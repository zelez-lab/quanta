#![cfg(feature = "render")]
//! Regression: the Vulkan render encoder must NOT create a fresh
//! VkSampler per textured draw per frame.
//!
//! Before the per-device sampler cache, every `.sampler(slot, desc)` in a
//! render pass called `vkCreateSampler` and never destroyed it — one
//! sampler leaked per textured draw per frame, exhausting the device's
//! `maxSamplerAllocationCount` pool (65,536 on v3dv) within minutes, after
//! which every glyph/textured draw failed. This drives a long textured-
//! draw loop and asserts the render-path sampler cache is bounded by the
//! number of DISTINCT descriptors, not the draw count.
//!
//! The `render_samplers` registry count is Vulkan-only (0 on other
//! backends). The exact-count assertions therefore run only when the
//! active backend reports a render sampler cache; on Metal/others the test
//! still exercises the draw loop end to end (a leak there would surface as
//! a device error), and the count checks note-skip. Lavapipe in CI is the
//! backend where the exact bound is proven.

use quanta::RenderGpu;

use quanta::render_pass::ColorTarget;
use quanta::{Color, FieldUsage, Filter, Format, LoadOp, SamplerDesc, StoreOp};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::vertex]
fn uv_vertex(pos: Vec3, uv: Vec2) -> Vec4 {
    Vec4::new(pos.x, pos.y, 0.0, 1.0)
}

#[quanta::fragment]
fn textured_frag(uv: Vec2) -> Vec4 {
    sample(0, uv)
}

fn pos_uv_layout() -> Vec<quanta::VertexLayout> {
    vec![quanta::VertexLayout {
        stride: 20,
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
                format: quanta::AttributeFormat::Float2,
            },
        ],
    }]
}

/// Render one textured-quad pass with the given sampler descriptor.
fn draw_textured(
    gpu: &quanta::Gpu,
    pipeline: &quanta::Pipeline,
    vb: &quanta::Field<f32>,
    tex: &quanta::Texture,
    target: &quanta::Texture,
    sampler: SamplerDesc,
    w: u32,
    h: u32,
) {
    let mut pulse = gpu
        .render(target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(pipeline)
        .vertices(0, vb)
        .texture(0, tex)
        .sampler(0, sampler)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
}

#[test]
fn sampler_cache_dedups_across_draws_and_frames() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if UV_VERTEX_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || TEXTURED_FRAG_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary for this vendor");
        return;
    }

    let layouts = pos_uv_layout();
    let pipeline = gpu
        .pipeline(
            &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
                vertex: &UV_VERTEX_SHADER,
                fragment: &TEXTURED_FRAG_SHADER,
            })
            .with_entries(
                UV_VERTEX_SHADER.entry_point,
                TEXTURED_FRAG_SHADER.entry_point,
            )
            .with_color_formats(vec![Format::RGBA8])
            .with_vertex_layouts(&layouts)
            .with_blend(quanta::BlendState::NONE),
        )
        .expect("pipeline creation");

    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(2, 2, Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ),
        )
        .expect("texture");
    tex.write(&[255u8; 16]).expect("tex write");

    #[rustfmt::skip]
    let verts: [f32; 30] = [
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0, -1.0, 0.0,  1.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0,  1.0, 0.0,  0.0, 1.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    let w = 8u32;
    let h = 8u32;
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    // The single descriptor reused for the whole loop. A distinct field of
    // ANY of its members would key a different cache entry.
    let desc_a = SamplerDesc::default().with_filters(Filter::Linear, Filter::Linear);

    // 200 textured draws (each its own pass = its own "frame") with the
    // SAME descriptor. Pre-cache leak: 200 live samplers. With the cache:
    // exactly one.
    for _ in 0..200 {
        draw_textured(&gpu, &pipeline, &vb, &tex, &target, desc_a, w, h);
    }

    let after_same = gpu.debug_registry_counts().render_samplers;
    if after_same == 0 {
        eprintln!(
            "note: backend does not report a render sampler cache \
             (non-Vulkan) — draw loop exercised, exact-count assertions skipped"
        );
        return;
    }
    assert_eq!(
        after_same, 1,
        "200 draws with one SamplerDesc must cache exactly ONE sampler \
         (found {after_same}); a per-draw create would leak"
    );

    // A DISTINCT descriptor (nearest filters) must add exactly one more.
    let desc_b = SamplerDesc::default().with_filters(Filter::Nearest, Filter::Nearest);
    for _ in 0..50 {
        draw_textured(&gpu, &pipeline, &vb, &tex, &target, desc_b, w, h);
    }
    let after_two = gpu.debug_registry_counts().render_samplers;
    assert_eq!(
        after_two, 2,
        "a second distinct SamplerDesc must add exactly one cache entry \
         (found {after_two})"
    );

    // Re-drawing with the first descriptor must NOT grow the cache.
    for _ in 0..50 {
        draw_textured(&gpu, &pipeline, &vb, &tex, &target, desc_a, w, h);
    }
    let after_reuse = gpu.debug_registry_counts().render_samplers;
    assert_eq!(
        after_reuse, 2,
        "re-drawing with a cached descriptor must not create a new sampler \
         (found {after_reuse})"
    );
}
