#![cfg(feature = "render")]
//! Cross-backend contract: a texture RENDERED INTO in one pass must be
//! SAMPLEABLE in a following pass — the render-to-texture-then-sample
//! shape every offscreen/effect/composite consumer drives.
//!
//! The trap this pins down (Vulkan): the render sample-bind path
//! (`write_render_descriptors`) writes `SHADER_READ_ONLY_OPTIMAL` into
//! every combined-image-sampler descriptor but emits NO barrier — it
//! assumed the image already arrived in that layout. That holds for an
//! uploaded texture or a `resolve_texture` output, but a texture just
//! rendered into sits in `COLOR_ATTACHMENT_OPTIMAL`; sampling it then is
//! a layout mismatch (`VUID-vkCmdDraw-None-09600`) that device-loses
//! some drivers (Intel Iris Xe). Metal has no explicit layouts (hazard
//! tracking is automatic), so this is green there and was the Vulkan-
//! only gap. Note the MSAA `resolve_texture`-then-sample path is covered
//! separately by `vulkan_msaa_resolve_sample`; THIS test is the
//! no-resolve path, which nothing else exercised.

use quanta::RenderGpu;

use quanta::render_pass::ColorTarget;
use quanta::{Color, FieldUsage, Filter, Format, LoadOp, SamplerDesc, StoreOp};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::vertex]
fn fill_vertex(pos: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, 0.0, 1.0)
}

#[quanta::fragment]
fn fill_green() -> Vec4 {
    Vec4::new(0.0, 1.0, 0.0, 1.0)
}

#[quanta::vertex]
fn uv_vertex(pos: Vec3, uv: Vec2) -> Vec4 {
    Vec4::new(pos.x, pos.y, 0.0, 1.0)
}

#[quanta::fragment]
fn textured_frag(uv: Vec2) -> Vec4 {
    sample(0, uv)
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

fn pixel_at(pixels: &[u8], w: u32, x: u32, y: u32) -> (u8, u8, u8, u8) {
    let i = ((y * w + x) * 4) as usize;
    (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3])
}

#[test]
fn render_into_texture_then_sample_it() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let vendor = gpu.caps().vendor;
    if FILL_VERTEX_SHADER.for_vendor(vendor).is_none()
        || FILL_GREEN_SHADER.for_vendor(vendor).is_none()
        || UV_VERTEX_SHADER.for_vendor(vendor).is_none()
        || TEXTURED_FRAG_SHADER.for_vendor(vendor).is_none()
    {
        eprintln!("SKIP: no shader binary for this vendor");
        return;
    }

    let w = 32u32;
    let h = 32u32;

    #[rustfmt::skip]
    let quad: [f32; 18] = [
        -1.0, -1.0, 0.0,
         1.0, -1.0, 0.0,
         1.0,  1.0, 0.0,
        -1.0, -1.0, 0.0,
         1.0,  1.0, 0.0,
        -1.0,  1.0, 0.0,
    ];
    let quad_vb: quanta::Field<f32> = gpu
        .field_with_usage(quad.len(), FieldUsage::default_render())
        .unwrap();
    quad_vb.write(&quad).unwrap();

    // ── 1. Render a solid green quad into `src` (1x render target,
    //       no resolve). It ends in COLOR_ATTACHMENT_OPTIMAL. ──────────
    let fill_pipeline = gpu
        .pipeline(
            &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
                vertex: &FILL_VERTEX_SHADER,
                fragment: &FILL_GREEN_SHADER,
            })
            .with_entries(
                FILL_VERTEX_SHADER.entry_point,
                FILL_GREEN_SHADER.entry_point,
            )
            .with_color_formats(vec![Format::RGBA8])
            .with_vertex_layouts(&pos_layout())
            .with_blend(quanta::BlendState::NONE),
        )
        .expect("fill pipeline");

    // `render_target()` carries RENDER_TARGET | SHADER_READ, so `src` is
    // sampleable — the point is the LAYOUT it's left in, not its usage.
    let src = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let mut pulse = gpu
        .render(&src)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&src)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&fill_pipeline)
        .vertices(0, &quad_vb)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    // ── 2. Sample `src` in a follow-up pass into `dst`. Without the
    //       COLOR_ATTACHMENT → SHADER_READ_ONLY transition on the
    //       sample-bind path, this mismatches the descriptor layout. ──
    let sample_pipeline = gpu
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
            .with_vertex_layouts(&pos_uv_layout())
            .with_blend(quanta::BlendState::NONE),
        )
        .expect("sample pipeline");

    #[rustfmt::skip]
    let uv_quad: [f32; 30] = [
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0, -1.0, 0.0,  1.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0,  1.0, 0.0,  0.0, 1.0,
    ];
    let uv_vb: quanta::Field<f32> = gpu
        .field_with_usage(uv_quad.len(), FieldUsage::default_render())
        .unwrap();
    uv_vb.write(&uv_quad).unwrap();

    let dst = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let mut pulse = gpu
        .render(&dst)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&dst)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&sample_pipeline)
        .vertices(0, &uv_vb)
        .texture(0, &src)
        .sampler(
            0,
            SamplerDesc::default().with_filters(Filter::Nearest, Filter::Nearest),
        )
        .draw(6)
        .pulse()
        .expect("sampling a rendered-into texture must succeed, not device-lose");
    pulse.wait().unwrap();

    // The sampled green must land in `dst` — proves the rendered-into
    // source was legibly sampled (green channel high, red/blue low).
    let pixels = dst.read().unwrap();
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    assert!(
        g > 200 && r < 60 && b < 60,
        "sampled rendered-into texture must carry its green color, got ({r},{g},{b})"
    );
}
