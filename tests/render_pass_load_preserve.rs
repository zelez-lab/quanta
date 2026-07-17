#![cfg(feature = "render")]
//! Cross-backend load-op parity: a pass that declares `LoadOp::Load`
//! must PRESERVE the target's previous contents — the multi-pass frame
//! contract every retained-mode consumer builds on (render the scene,
//! then composite a layer, then overlay glyphs, each in its own pass).
//!
//! The trap this pins down: the Vulkan driver began every
//! pipeline-bound pass with the pipeline's BAKED render pass, whose
//! load op is hardcoded CLEAR — so a `LoadOp::Load` pass wiped the
//! target to the fallback clear value (0,0,0,1). First observed as
//! dija's headless render-probe reading back solid (0,0,0,255) frames
//! on Windows/Iris Xe: every pass before the last was erased, while
//! the single-pass swapchain route (and Metal, which maps
//! `LoadOp::Load` to MTLLoadActionLoad) rendered correctly.

use quanta::RenderGpu;

use quanta::render_pass::ColorTarget;
use quanta::{Color, FieldUsage, Format, LoadOp, StoreOp};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
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

fn pipeline_for(
    gpu: &quanta::Gpu,
    frag: &'static quanta::ShaderBinary,
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
        .with_blend(quanta::BlendState::NONE),
    )
    .ok()
}

#[test]
fn second_pass_load_preserves_first_pass_output() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_WHITE_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary for this vendor");
        return;
    }

    let white = pipeline_for(&gpu, &SOLID_WHITE_SHADER).expect("white pipeline");
    let red = pipeline_for(&gpu, &SOLID_RED_SHADER).expect("red pipeline");

    // Quad 1 fills the LEFT half; quad 2 the bottom-right quadrant.
    // NDC y is up; pixel y is down — quadrant checks below only use
    // left/right halves and the horizontal mid-line to stay
    // orientation-proof.
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
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    // Pass 1: clear to mid-gray, draw the white left half.
    let mut p1 = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.5, 0.5, 0.5, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&white)
        .vertices(0, &vb1)
        .draw(6)
        .pulse()
        .expect("pass 1");
    p1.wait().unwrap();

    // Pass 2: LOAD (no clear), draw a red strip on the far right.
    let mut p2 = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Load)
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&red)
        .vertices(0, &vb2)
        .draw(6)
        .pulse()
        .expect("pass 2");
    p2.wait().unwrap();

    let pixels = target.read().unwrap();

    // Pass 1's white left half must SURVIVE pass 2.
    let (r, g, b, _) = pixel_at(&pixels, w, w / 4, h / 2);
    assert!(
        r > 200 && g > 200 && b > 200,
        "pass 1 output must survive a LoadOp::Load pass 2, got ({r},{g},{b}) — \
         a (0,0,0) read here means the second pass CLEARED instead of loading"
    );

    // Pass 1's clear color must survive where neither quad drew
    // (between the quads: x in (mid, 0.75·w)).
    let (r, g, b, _) = pixel_at(&pixels, w, (w * 5) / 8, h / 2);
    assert!(
        (100..=200).contains(&r) && (100..=200).contains(&g) && (100..=200).contains(&b),
        "pass 1 clear color must survive a LoadOp::Load pass 2, got ({r},{g},{b})"
    );

    // Pass 2's red strip landed.
    let (r, g, b, _) = pixel_at(&pixels, w, (w * 15) / 16, h / 2);
    assert!(
        r > 200 && g < 50 && b < 50,
        "pass 2 draw must land, got ({r},{g},{b})"
    );
}
