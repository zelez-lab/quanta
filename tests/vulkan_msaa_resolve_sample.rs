#![cfg(feature = "render")]
//! MSAA render -> resolve -> sample-the-resolve, the shape dija's layer
//! path drives. On lavapipe with the Khronos validation layers this is the
//! regression net for the barrier/resolve correctness fix: the resolve now
//! routes its image transitions through each texture's TRACKED layout
//! instead of hardcoding COLOR_ATTACHMENT_OPTIMAL as the source oldLayout,
//! which previously mismatched (VUID-VkImageMemoryBarrier-oldLayout-01211)
//! when the resolve source had last been sampled. Resolve targets are
//! created with TRANSFER_DST usage, satisfying
//! VUID-vkCmdResolveImage-dstImage-06764.
//!
//! The value assertion (the resolved image, once sampled, carries the
//! rendered color) doubles as a functional check on any backend that
//! supports resolve; validation-layer cleanliness is what the Vulkan CI
//! lane adds on top.

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
fn fill_blue() -> Vec4 {
    Vec4::new(0.0, 0.0, 1.0, 1.0)
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
fn msaa_resolve_then_sample_the_resolve() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let vendor = gpu.caps().vendor;
    if FILL_VERTEX_SHADER.for_vendor(vendor).is_none()
        || FILL_BLUE_SHADER.for_vendor(vendor).is_none()
        || UV_VERTEX_SHADER.for_vendor(vendor).is_none()
        || TEXTURED_FRAG_SHADER.for_vendor(vendor).is_none()
    {
        eprintln!("SKIP: no shader binary for this vendor");
        return;
    }

    let w = 32u32;
    let h = 32u32;

    // ── 1. Render a solid blue quad into a 4x MSAA target ──────────────
    let fill_pipeline = gpu
        .pipeline(
            &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
                vertex: &FILL_VERTEX_SHADER,
                fragment: &FILL_BLUE_SHADER,
            })
            .with_entries(FILL_VERTEX_SHADER.entry_point, FILL_BLUE_SHADER.entry_point)
            .with_color_formats(vec![Format::RGBA8])
            .with_vertex_layouts(&pos_layout())
            .with_sample_count(4)
            .with_blend(quanta::BlendState::NONE),
        )
        .expect("msaa fill pipeline");

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

    let msaa = gpu.msaa_target(w, h, Format::RGBA8, 4).unwrap();
    // The resolve destination is ALSO sampled afterwards, so it needs
    // SHADER_READ. (It always carries TRANSFER_DST for the resolve.)
    let resolved = gpu
        .create_texture(&quanta::TextureDesc::new(w, h, Format::RGBA8).with_usage(
            quanta::TextureUsage::RENDER_TARGET.union(quanta::TextureUsage::SHADER_READ),
        ))
        .expect("resolve target");

    let mut pulse = gpu
        .render(&msaa)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&msaa)
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

    // ── 2. Resolve MSAA -> single-sample ──────────────────────────────
    if let Err(e) = gpu.resolve_texture(&msaa, &resolved) {
        eprintln!("SKIP: resolve_texture not supported: {e}");
        return;
    }

    // ── 3. Sample the resolved image in a follow-up textured draw ──────
    // This is the transition that used to mismatch: the resolve source and
    // destination end in SHADER_READ_ONLY, and sampling `resolved` must see
    // that tracked layout rather than a stale assumption.
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

    let final_target = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let mut pulse = gpu
        .render(&final_target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&final_target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&sample_pipeline)
        .vertices(0, &uv_vb)
        .texture(0, &resolved)
        .sampler(
            0,
            SamplerDesc::default().with_filters(Filter::Nearest, Filter::Nearest),
        )
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    let pixels = final_target.read().unwrap();
    // The whole chain (MSAA blue -> resolve -> sample) must land blue in
    // the center, proving the resolved image was both produced and
    // sampleable through its tracked layout.
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    eprintln!("resolved-then-sampled center: rgba({r},{g},{b})");
    assert!(
        b > 200 && r < 60 && g < 60,
        "sampling the resolved MSAA image should yield blue, got ({r},{g},{b})"
    );
}
