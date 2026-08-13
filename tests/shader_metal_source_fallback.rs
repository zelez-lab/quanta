#![cfg(feature = "render")]
//! The off-Apple-host bundle story, exercised on real Metal: a
//! `ShaderBinary` whose `metallib` field carries MSL SOURCE bytes
//! (what the compiler ships when the Apple toolchain is absent) must
//! build a pipeline and draw — the Metal driver sniffs the `MTLB`
//! magic and compiles non-MTLB bytes as source at pipeline creation.

use quanta::RenderGpu;
use quanta::render_pass::ColorTarget;
use quanta::{Color, FieldUsage, Format, LoadOp, StoreOp};

const SRC_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VIn {
    float3 pos [[attribute(0)]];
};

vertex float4 src_vert(VIn in [[stage_in]]) {
    return float4(in.pos, 1.0);
}

fragment float4 src_frag() {
    return float4(0.3, 0.7, 0.2, 1.0);
}
"#;

static SRC_VERT: quanta::ShaderBinary = quanta::ShaderBinary {
    spirv: None,
    metallib: Some(SRC_MSL.as_bytes()),
    metallib_ios: None,
    metallib_ios_sim: None,
    wgsl: None,
    entry_point: "src_vert",
    stage: quanta::ShaderStage::Vertex,
};

static SRC_FRAG: quanta::ShaderBinary = quanta::ShaderBinary {
    spirv: None,
    metallib: Some(SRC_MSL.as_bytes()),
    metallib_ios: None,
    metallib_ios_sim: None,
    wgsl: None,
    entry_point: "src_frag",
    stage: quanta::ShaderStage::Fragment,
};

#[rustfmt::skip]
const FULLSCREEN_POS: [f32; 18] = [
    -1.0, -1.0, 0.0,
     1.0, -1.0, 0.0,
     1.0,  1.0, 0.0,
    -1.0, -1.0, 0.0,
     1.0,  1.0, 0.0,
    -1.0,  1.0, 0.0,
];

#[test]
fn msl_source_in_the_metallib_field_builds_and_draws() {
    let Some(gpu) = quanta::init().ok() else {
        return;
    };
    if gpu.caps().vendor != quanta::Vendor::Apple {
        eprintln!("SKIP: Metal-only (the sniff lives in the Metal binary loader)");
        return;
    }

    let layouts = vec![quanta::VertexLayout {
        stride: 12,
        step: quanta::StepMode::Vertex,
        attributes: vec![quanta::VertexAttribute {
            location: 0,
            offset: 0,
            format: quanta::AttributeFormat::Float3,
        }],
    }];
    let pipe = gpu
        .pipeline(
            &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
                vertex: &SRC_VERT,
                fragment: &SRC_FRAG,
            })
            .with_entries(SRC_VERT.entry_point, SRC_FRAG.entry_point)
            .with_color_formats(vec![Format::RGBA8])
            .with_vertex_layouts(&layouts)
            .with_blend(quanta::BlendState::NONE),
        )
        .expect("pipeline builds from MSL source in the metallib field");

    let vb: quanta::Field<f32> = gpu
        .field_with_usage(FULLSCREEN_POS.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&FULLSCREEN_POS).unwrap();

    let (w, h) = (4u32, 4u32);
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
        .pipeline(&pipe)
        .vertices(0, &vb)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
    let px = target.read().unwrap();
    // 0.3/0.7/0.2 → 77/178/51.
    let i = ((2 * w + 2) * 4) as usize;
    let (r, g, b) = (px[i], px[i + 1], px[i + 2]);
    assert!(
        r.abs_diff(77) <= 2 && g.abs_diff(178) <= 2 && b.abs_diff(51) <= 2,
        "source-compiled shader drew wrong color: ({r},{g},{b})"
    );
}
