#![cfg(feature = "render")]
//! Builder-managed MSAA (`RenderBuilder::msaa`) — the backend-owned
//! MSAA lifecycle dija used to hand-manage (~450 LOC of pooled
//! intermediate + Store-across-sub-passes + trailing resolve).
//!
//! The core contract under test: samples stored by one `.msaa(n)` pass
//! SURVIVE into the next `.msaa(n)` pass over the same target (the
//! pooled intermediate is found again and LOADed), and the pass that
//! ends with `.msaa_resolve()` resolves the accumulated samples into
//! the original single-sample target. Plus the guard rails: dangling
//! `.msaa_resolve()`, pipeline/sample-count mismatch (the existing
//! encode-time validation, fed by the intermediate's sample count),
//! `.msaa()` on an already-multisampled target, and the
//! `color_targets()` conflict.
//!
//! Also here: `render_into` (#13), the closure form of `render` for
//! callers whose target borrow collides with `&mut self`.

use quanta::RenderGpu;

use quanta::{Color, FieldUsage, Format, QuantaErrorKind};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// A GPU whose render path is live (`render_begin` succeeds) — the CPU
/// software device has no rasterizer and must skip these tests.
fn try_render_gpu() -> Option<(quanta::Gpu, quanta::Texture)> {
    let gpu = try_gpu()?;
    let probe = gpu.render_target(8, 8, Format::RGBA8).ok()?;
    if gpu.render(&probe).is_err() {
        eprintln!("skipping: no render path on this device");
        return None;
    }
    Some((gpu, probe))
}

#[quanta::vertex]
fn passthrough_vertex(pos: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, 0.0, 1.0)
}

#[quanta::fragment]
fn solid_white() -> Vec4 {
    Vec4::new(1.0, 1.0, 1.0, 1.0)
}

#[quanta::fragment]
fn solid_red() -> Vec4 {
    Vec4::new(1.0, 0.0, 0.0, 1.0)
}

fn pos_layout() -> Vec<quanta::VertexLayout> {
    vec![quanta::VertexLayout {
        stride: 12,
        step: quanta::StepMode::Vertex,
        attributes: vec![quanta::VertexAttribute {
            location: 0,
            offset: 0,
            format: quanta::AttributeFormat::Float3,
        }],
    }]
}

/// Two triangles covering the NDC rect [x0,x1]×[y0,y1].
#[rustfmt::skip]
fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> [f32; 18] {
    [
        x0, y0, 0.0,
        x1, y0, 0.0,
        x1, y1, 0.0,
        x0, y0, 0.0,
        x1, y1, 0.0,
        x0, y1, 0.0,
    ]
}

fn pixel_at(pixels: &[u8], w: u32, x: u32, y: u32) -> (u8, u8, u8, u8) {
    let i = ((y * w + x) * 4) as usize;
    (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3])
}

/// `expect_err` for pulse results (`Pulse` has no `Debug`).
fn expect_pulse_err(
    result: Result<quanta::Pulse, quanta::QuantaError>,
    what: &str,
) -> quanta::QuantaError {
    match result {
        Ok(_) => panic!("{what} must fail"),
        Err(e) => e,
    }
}

fn pipeline_at(
    gpu: &quanta::Gpu,
    frag: &'static quanta::ShaderBinary,
    sample_count: u32,
) -> Option<quanta::Pipeline> {
    let layouts = pos_layout();
    gpu.pipeline(
        &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
            vertex: &PASSTHROUGH_VERTEX_SHADER,
            fragment: frag,
        })
        .with_entries(PASSTHROUGH_VERTEX_SHADER.entry_point, frag.entry_point)
        .with_color_formats(vec![Format::RGBA8])
        .with_vertex_layouts(&layouts)
        .with_sample_count(sample_count)
        .with_blend(quanta::BlendState::NONE),
    )
    .ok()
}

fn shaders_available(gpu: &quanta::Gpu) -> bool {
    let vendor = gpu.caps().vendor;
    if PASSTHROUGH_VERTEX_SHADER.for_vendor(vendor).is_none()
        || SOLID_WHITE_SHADER.for_vendor(vendor).is_none()
        || SOLID_RED_SHADER.for_vendor(vendor).is_none()
    {
        eprintln!("SKIP: no shader binary for this vendor");
        return false;
    }
    true
}

/// The design-sketch flow, end to end: pass 1 clears the pooled
/// intermediate and draws; pass 2 finds the SAME intermediate, LOADs
/// it (samples preserved), draws more, and subpass-resolves into the
/// target. The readback must show BOTH passes' content — the
/// sample-preservation contract dija needs.
#[test]
fn two_pass_msaa_preserves_samples_and_resolves() {
    let Some((gpu, _probe)) = try_render_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !shaders_available(&gpu) {
        return;
    }

    let Some(white4x) = pipeline_at(&gpu, &SOLID_WHITE_SHADER, 4) else {
        eprintln!("SKIP: 4x pipeline creation failed");
        return;
    };
    let Some(red4x) = pipeline_at(&gpu, &SOLID_RED_SHADER, 4) else {
        eprintln!("SKIP: 4x pipeline creation failed");
        return;
    };

    // Quad 1 fills the LEFT half; quad 2 a strip on the far right.
    // Assertions stick to left/right halves + the horizontal mid-line
    // to stay orientation-proof.
    let vb1: quanta::Field<f32> = gpu
        .field_with_usage(18, FieldUsage::default_render())
        .unwrap();
    vb1.write(&quad(-1.0, -1.0, 0.0, 1.0)).unwrap();
    let vb2: quanta::Field<f32> = gpu
        .field_with_usage(18, FieldUsage::default_render())
        .unwrap();
    vb2.write(&quad(0.5, -1.0, 1.0, 1.0)).unwrap();

    let w = 64u32;
    let h = 64u32;
    // The RESOLVE destination: 1x, sampleable.
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    // Pass 1: clear the MSAA intermediate to mid-gray, draw the white
    // left half. Ends with Store — no resolve, samples kept.
    let mut p1 = gpu
        .render(&target)
        .unwrap()
        .msaa(4)
        .clear(Color::rgba(0.5, 0.5, 0.5, 1.0))
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&white4x)
        .vertices(0, &vb1)
        .draw(6)
        .pulse()
        .expect("msaa pass 1");
    p1.wait().unwrap();

    // Pass 2: SAME pooled intermediate, LOADed — pass 1's samples must
    // survive. Draw the red strip, then resolve into `target`.
    let mut p2 = gpu
        .render(&target)
        .unwrap()
        .msaa(4)
        .load()
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&red4x)
        .vertices(0, &vb2)
        .draw(6)
        .msaa_resolve()
        .pulse()
        .expect("msaa pass 2 + resolve");
    p2.wait().unwrap();

    let pixels = target.read().unwrap();

    // Pass 1's white left half survived pass 2's load.
    let (r, g, b, _) = pixel_at(&pixels, w, w / 4, h / 2);
    assert!(
        r > 200 && g > 200 && b > 200,
        "pass 1 samples must survive into the loading pass 2 and resolve, got \
         ({r},{g},{b}) — a (0,0,0) or gray-free read here means the intermediate \
         was re-cleared or a fresh one was created instead of the pooled reuse"
    );

    // Pass 1's clear color survived where neither quad drew.
    let (r, g, b, _) = pixel_at(&pixels, w, (w * 5) / 8, h / 2);
    assert!(
        (100..=200).contains(&r) && (100..=200).contains(&g) && (100..=200).contains(&b),
        "pass 1's clear must survive pass 2's load + resolve, got ({r},{g},{b})"
    );

    // Pass 2's red strip landed and resolved.
    let (r, g, b, _) = pixel_at(&pixels, w, (w * 15) / 16, h / 2);
    assert!(
        r > 200 && g < 50 && b < 50,
        "pass 2 draw must land in the resolved target, got ({r},{g},{b})"
    );
}

/// `.msaa_resolve()` on a pass without `.msaa(n)` is a dangling
/// resolve — InvalidParam at pulse().
#[test]
fn msaa_resolve_without_msaa_errors() {
    let Some((gpu, target)) = try_render_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let err = expect_pulse_err(
        gpu.render(&target).unwrap().msaa_resolve().pulse(),
        "msaa_resolve() without msaa(n)",
    );
    assert!(
        matches!(err.kind, QuantaErrorKind::InvalidParam(_)),
        "expected InvalidParam, got {err:?}"
    );
    assert!(
        err.to_string().contains("msaa_resolve() without msaa"),
        "unexpected message: {err}"
    );
}

/// `.load()` is an MSAA-pass marker; on a plain pass it is
/// InvalidParam (the message points at ColorTarget::with_load_op).
#[test]
fn load_without_msaa_errors() {
    let Some((gpu, target)) = try_render_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let err = expect_pulse_err(
        gpu.render(&target).unwrap().load().pulse(),
        "load() without msaa(n)",
    );
    assert!(
        matches!(err.kind, QuantaErrorKind::InvalidParam(_)),
        "expected InvalidParam, got {err:?}"
    );
}

/// A single-sample pipeline bound under `.msaa(4)` must trip the
/// existing encode-time sample-count validation: the pooled
/// intermediate carries sample count 4, the pipeline declares 1.
#[test]
fn msaa_pipeline_sample_mismatch_errors() {
    let Some((gpu, _probe)) = try_render_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !shaders_available(&gpu) {
        return;
    }
    let Some(white1x) = pipeline_at(&gpu, &SOLID_WHITE_SHADER, 1) else {
        eprintln!("SKIP: pipeline creation failed");
        return;
    };
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(18, FieldUsage::default_render())
        .unwrap();
    vb.write(&quad(-1.0, -1.0, 1.0, 1.0)).unwrap();

    let target = gpu.render_target(32, 32, Format::RGBA8).unwrap();
    let err = expect_pulse_err(
        gpu.render(&target)
            .unwrap()
            .msaa(4)
            .clear(Color::BLACK)
            .pipeline(&white1x)
            .vertices(0, &vb)
            .draw(6)
            .pulse(),
        "a 1x pipeline under .msaa(4)",
    );
    assert!(
        matches!(err.kind, QuantaErrorKind::InvalidParam(_)),
        "expected InvalidParam, got {err:?}"
    );
    assert!(
        err.to_string().contains("sample-count mismatch"),
        "expected the encode-time sample-count validation to fire, got: {err}"
    );
}

/// `.msaa(n)` needs the single-sample resolve destination as its
/// target; handing it an already-multisampled texture is InvalidParam.
#[test]
fn msaa_on_multisampled_target_errors() {
    let Some((gpu, _probe)) = try_render_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let msaa_target = gpu.msaa_target(32, 32, Format::RGBA8, 4).unwrap();
    let err = expect_pulse_err(
        gpu.render(&msaa_target).unwrap().msaa(4).pulse(),
        ".msaa() on a multisampled target",
    );
    assert!(
        matches!(err.kind, QuantaErrorKind::InvalidParam(_)),
        "expected InvalidParam, got {err:?}"
    );
}

/// `.msaa(n)` owns the pass's color attachment — combining it with an
/// explicit `.color_targets(..)` is InvalidParam.
#[test]
fn msaa_conflicts_with_explicit_color_targets() {
    let Some((gpu, target)) = try_render_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let err = expect_pulse_err(
        gpu.render(&target)
            .unwrap()
            .msaa(4)
            .color_targets(vec![quanta::render_pass::ColorTarget::new(&target)])
            .pulse(),
        ".msaa() + .color_targets()",
    );
    assert!(
        matches!(err.kind, QuantaErrorKind::InvalidParam(_)),
        "expected InvalidParam, got {err:?}"
    );
}

/// Changing `n` between passes over the same target evicts and
/// recreates the pooled intermediate: a 4x clear+resolve followed by a
/// 2x clear+resolve must land the second pass's color (through a fresh
/// 2x intermediate), not stale 4x content.
#[test]
fn msaa_sample_count_change_evicts_and_recreates() {
    let Some((gpu, _probe)) = try_render_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !shaders_available(&gpu) {
        return;
    }
    let Some(white4x) = pipeline_at(&gpu, &SOLID_WHITE_SHADER, 4) else {
        eprintln!("SKIP: 4x pipeline creation failed");
        return;
    };
    let Some(red2x) = pipeline_at(&gpu, &SOLID_RED_SHADER, 2) else {
        eprintln!("SKIP: 2x pipeline creation failed");
        return;
    };
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(18, FieldUsage::default_render())
        .unwrap();
    vb.write(&quad(-1.0, -1.0, 1.0, 1.0)).unwrap();

    let w = 32u32;
    let h = 32u32;
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let mut p1 = gpu
        .render(&target)
        .unwrap()
        .msaa(4)
        .clear(Color::BLACK)
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&white4x)
        .vertices(0, &vb)
        .draw(6)
        .msaa_resolve()
        .pulse()
        .expect("4x pass");
    p1.wait().unwrap();

    // Same target, different sample count: the 4x intermediate is
    // evicted (it is idle — p1 was waited) and a 2x one created.
    let mut p2 = gpu
        .render(&target)
        .unwrap()
        .msaa(2)
        .clear(Color::BLACK)
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&red2x)
        .vertices(0, &vb)
        .draw(6)
        .msaa_resolve()
        .pulse()
        .expect("2x pass after 4x");
    p2.wait().unwrap();

    let pixels = target.read().unwrap();
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    assert!(
        r > 200 && g < 50 && b < 50,
        "the 2x pass must render through a recreated intermediate, got ({r},{g},{b})"
    );
}

/// `render_into` (#13): the closure form releases the target borrow
/// when it returns, so a call site holding `&mut self` state can draw
/// without laundering the borrow through a raw pointer. Draw a plain
/// quad through it and read the result back.
#[test]
fn render_into_closure_draws() {
    let Some((gpu, _probe)) = try_render_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !shaders_available(&gpu) {
        return;
    }
    let Some(white) = pipeline_at(&gpu, &SOLID_WHITE_SHADER, 1) else {
        eprintln!("SKIP: pipeline creation failed");
        return;
    };

    // The borrow-collision shape from dija: the target and the dirty
    // flag live in the same struct, and the method mutates the flag
    // around the draw.
    struct Layer {
        target: quanta::Texture,
        dirty: bool,
    }
    let w = 32u32;
    let h = 32u32;
    let mut layer = Layer {
        target: gpu.render_target(w, h, Format::RGBA8).unwrap(),
        dirty: true,
    };

    let vb: quanta::Field<f32> = gpu
        .field_with_usage(18, FieldUsage::default_render())
        .unwrap();
    vb.write(&quad(-1.0, -1.0, 1.0, 1.0)).unwrap();

    let mut pulse = gpu
        .render_into(&layer.target, |b| {
            b.clear(Color::BLACK)
                .viewport(0.0, 0.0, w as f32, h as f32)
                .pipeline(&white)
                .vertices(0, &vb)
                .draw(6)
                .pulse()
        })
        .expect("render_into");
    layer.dirty = false; // the target borrow ended with render_into
    pulse.wait().unwrap();

    assert!(!layer.dirty);
    let pixels = layer.target.read().unwrap();
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    assert!(
        r > 200 && g > 200 && b > 200,
        "render_into draw must land, got ({r},{g},{b})"
    );
}
