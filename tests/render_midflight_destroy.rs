#![cfg(feature = "render")]
//! Mid-flight destroys: render wrappers dropped between `pulse()` and
//! the wait.
//!
//! Render submissions are asynchronous — `pulse()` returns a live
//! `Pulse` while the GPU still executes the pass — so a `Pipeline` or
//! occlusion-query drop can reach the driver's destroy while the
//! submitted command buffer references the resource. Vulkan must
//! defer those destroys behind the submission serial (the retire
//! bin): destroying inline is VUID-vkDestroyPipeline-pipeline-00765 /
//! -vkDestroyQueryPool-queryPool-00793 territory. Metal covers the
//! same shape by refcounting — the command buffer retains what it
//! encodes. These tests pin the contract on every backend: the pass
//! outlives its wrappers, whether the pulse is waited or dropped.
//!
//! Requires a GPU; skips gracefully if none available.

use quanta::RenderGpu;

use quanta::render_pass::ColorTarget;
use quanta::{Color, FieldUsage, Format, LoadOp, StoreOp};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// ─── Shaders ────────────────────────────────────────────────────────────────

#[quanta::vertex]
fn midflight_vertex(pos: Vec3, color: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn midflight_red() -> Vec4 {
    Vec4::new(1.0, 0.0, 0.0, 1.0)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn pos_color_layout() -> Vec<quanta::VertexLayout> {
    vec![quanta::VertexLayout {
        stride: 24,
        step: quanta::StepMode::Vertex,
        attributes: vec![
            quanta::VertexAttribute {
                location: 0,
                offset: 0,
                format: quanta::AttributeFormat::Float3,
            },
            quanta::VertexAttribute {
                location: 1,
                offset: 12,
                format: quanta::AttributeFormat::Float3,
            },
        ],
    }]
}

fn make_pipeline(gpu: &quanta::Gpu, layouts: &[quanta::VertexLayout]) -> Option<quanta::Pipeline> {
    if MIDFLIGHT_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || MIDFLIGHT_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("skipping: no shader binary for this vendor");
        return None;
    }
    let desc = quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
        vertex: &MIDFLIGHT_VERTEX_SHADER,
        fragment: &MIDFLIGHT_RED_SHADER,
    })
    .with_entries(
        MIDFLIGHT_VERTEX_SHADER.entry_point,
        MIDFLIGHT_RED_SHADER.entry_point,
    )
    .with_color_formats(vec![Format::RGBA8])
    .with_vertex_layouts(layouts)
    .with_blend(quanta::BlendState::NONE);
    Some(gpu.pipeline(&desc).expect("pipeline creation"))
}

fn triangle_vb(gpu: &quanta::Gpu) -> quanta::Field<f32> {
    #[rustfmt::skip]
    let verts: [f32; 18] = [
         0.0, -0.5, 0.0,   1.0, 0.0, 0.0,
        -0.5,  0.5, 0.0,   0.0, 1.0, 0.0,
         0.5,  0.5, 0.0,   0.0, 0.0, 1.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .expect("vb");
    vb.write(&verts).expect("write vb");
    vb
}

fn pixel_at(pixels: &[u8], w: u32, x: u32, y: u32) -> (u8, u8, u8, u8) {
    let i = ((y * w + x) * 4) as usize;
    (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3])
}

fn assert_red_triangle(target: &quanta::Texture, w: u32, h: u32) {
    let pixels = target.read().unwrap();
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    assert!(r > 200, "center should be red (R={r})");
    assert!(g < 50 && b < 50, "center should not be green/blue");
}

// ─── Pipeline dropped between pulse() and wait() ────────────────────────────

#[test]
fn pipeline_dropped_between_pulse_and_wait() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let layouts = pos_color_layout();
    let Some(pipeline) = make_pipeline(&gpu, &layouts) else {
        return;
    };
    let vb = triangle_vb(&gpu);

    let (w, h) = (64u32, 64u32);
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .unwrap();

    // The submission is in flight; dropping the pipeline now must
    // defer the driver-side destroy behind it.
    drop(pipeline);
    pulse.wait().unwrap();

    assert_red_triangle(&target, w, h);
}

// ─── Pipeline AND pulse dropped, nothing ever waited ────────────────────────

#[test]
fn pipeline_and_pulse_dropped_unwaited() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let layouts = pos_color_layout();
    let Some(pipeline) = make_pipeline(&gpu, &layouts) else {
        return;
    };
    let vb = triangle_vb(&gpu);

    let (w, h) = (64u32, 64u32);
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let pulse = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .unwrap();

    // Drop the pipeline first (parks behind the in-flight
    // submission), then the pulse without ever waiting — the pulse's
    // deferred cleanup completes the fence and drains the park.
    drop(pipeline);
    drop(pulse);

    // The device must still be healthy: the same draw renders again
    // through a fresh pipeline.
    let Some(pipeline) = make_pipeline(&gpu, &layouts) else {
        return;
    };
    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
    assert_red_triangle(&target, w, h);
}

// ─── Query set dropped between pulse() and wait() ───────────────────────────

#[test]
fn query_set_dropped_between_pulse_and_wait() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let layouts = pos_color_layout();
    let Some(pipeline) = make_pipeline(&gpu, &layouts) else {
        return;
    };
    let vb = triangle_vb(&gpu);

    let query = match gpu.occlusion_query_create(1) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("skipping: occlusion queries not supported: {e}");
            return;
        }
    };

    let (w, h) = (64u32, 64u32);
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .begin_occlusion_query(&query, 0)
        .draw(3)
        .end_occlusion_query(&query, 0)
        .pulse()
        .unwrap();

    // The submitted pass still writes the query pool; dropping the
    // set now must defer the pool destroy behind the fence. The
    // result is never read — the drop-without-read shape.
    drop(query);
    pulse.wait().unwrap();

    assert_red_triangle(&target, w, h);
}
