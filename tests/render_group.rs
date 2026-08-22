#![cfg(feature = "render")]
//! `Gpu::render_group` — pooled offscreen layers.
//!
//! Each test proves one leg of the group contract: a group's contents
//! are visible to a LATER pass in the same frame with no host wait
//! (submission order + render-then-sample), groups nest (each is its
//! own pass), the pool reuses returned intermediates by driver handle,
//! and `.msaa(n)` inside a group resolves into the pooled layer.

use quanta::RenderGpu;
use quanta::render_pass::ColorTarget;
use quanta::{Color, FieldUsage, Format, LoadOp, StoreOp};
use quanta::{Vec2, Vec4};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[rustfmt::skip]
const FULLSCREEN_QUAD: [f32; 30] = [
    -1.0, -1.0, 0.0,  0.0, 0.0,
     1.0, -1.0, 0.0,  1.0, 0.0,
     1.0,  1.0, 0.0,  1.0, 1.0,
    -1.0, -1.0, 0.0,  0.0, 0.0,
     1.0,  1.0, 0.0,  1.0, 1.0,
    -1.0,  1.0, 0.0,  0.0, 1.0,
];

fn fullscreen_vb(gpu: &quanta::Gpu) -> quanta::Field<f32> {
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(FULLSCREEN_QUAD.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&FULLSCREEN_QUAD).unwrap();
    vb
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

fn pipeline(
    gpu: &quanta::Gpu,
    vert: &quanta::ShaderBinary,
    frag: &quanta::ShaderBinary,
) -> quanta::Pipeline {
    let layouts = pos_uv_layout();
    gpu.pipeline(
        &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
            vertex: vert,
            fragment: frag,
        })
        .with_entries(vert.entry_point, frag.entry_point)
        .with_color_formats(vec![Format::RGBA8])
        .with_vertex_layouts(&layouts)
        .with_blend(quanta::BlendState::NONE),
    )
    .expect("pipeline creation")
}

fn shaders_ready(gpu: &quanta::Gpu, bins: &[&quanta::ShaderBinary]) -> bool {
    bins.iter()
        .all(|b| b.for_artifact(gpu.artifact_kind()).is_some())
}

#[derive(quanta::Varyings)]
struct GroupVary {
    #[position]
    clip: Vec4,
    uv: Vec2,
}

#[quanta::vertex]
fn group_quad_vertex(pos: quanta::Vec3, uv: Vec2) -> GroupVary {
    GroupVary {
        clip: Vec4::new(pos.x, pos.y, 0.0, 1.0),
        uv,
    }
}

#[quanta::fragment]
fn solid_layer_frag() -> Vec4 {
    Vec4::new(0.2, 0.6, 1.0, 1.0)
}

#[quanta::fragment]
fn sample_layer_frag(s: GroupVary) -> Vec4 {
    sample(0, s.uv)
}

fn expect_rgb(pixels: &[u8], w: u32, x: u32, y: u32, want: (u8, u8, u8), which: &str) {
    let i = ((y * w + x) * 4) as usize;
    let (r, g, b) = (pixels[i], pixels[i + 1], pixels[i + 2]);
    assert!(
        r.abs_diff(want.0) <= 2 && g.abs_diff(want.1) <= 2 && b.abs_diff(want.2) <= 2,
        "{which} at ({x},{y}): expected {want:?}, got ({r},{g},{b})"
    );
}

/// Draw the fullscreen quad with `frag`'s pipeline into a builder,
/// binding `layer` at texture slot 0 when given.
fn draw_quad(
    b: quanta::RenderBuilder,
    pipe: &quanta::Pipeline,
    vb: &quanta::Field<f32>,
    layer: Option<&quanta::Texture>,
    (w, h): (u32, u32),
) -> Result<quanta::Pulse, quanta::QuantaError> {
    let mut b = b
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(pipe)
        .vertices(0, vb);
    if let Some(tex) = layer {
        b = b.texture(0, tex).sampler(
            0,
            quanta::SamplerDesc::default()
                .with_filters(quanta::Filter::Nearest, quanta::Filter::Nearest),
        );
    }
    b.draw(6).pulse()
}

#[test]
fn group_renders_and_parent_samples_same_frame() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(
        &gpu,
        &[
            &GROUP_QUAD_VERTEX_SHADER,
            &SOLID_LAYER_FRAG_SHADER,
            &SAMPLE_LAYER_FRAG_SHADER,
        ],
    ) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let solid = pipeline(&gpu, &GROUP_QUAD_VERTEX_SHADER, &SOLID_LAYER_FRAG_SHADER);
    let sampling = pipeline(&gpu, &GROUP_QUAD_VERTEX_SHADER, &SAMPLE_LAYER_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    let layer = gpu
        .render_group((32, 32), Format::RGBA8, |b| {
            draw_quad(
                b.clear(Color::rgba(0.0, 0.0, 0.0, 1.0)),
                &solid,
                &vb,
                None,
                (32, 32),
            )
        })
        .expect("group renders");

    let (w, h) = (8u32, 8u32);
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
        .pipeline(&sampling)
        .vertices(0, &vb)
        .texture(0, &layer)
        .sampler(
            0,
            quanta::SamplerDesc::default()
                .with_filters(quanta::Filter::Nearest, quanta::Filter::Nearest),
        )
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
    let px = target.read().unwrap();
    // 0.2/0.6/1.0 → 51/153/255.
    expect_rgb(&px, w, 2, 2, (51, 153, 255), "sampled layer");
    expect_rgb(&px, w, w - 1, h - 1, (51, 153, 255), "sampled layer corner");
}

#[test]
fn group_pool_reuses_by_driver_handle() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&GROUP_QUAD_VERTEX_SHADER, &SOLID_LAYER_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let solid = pipeline(&gpu, &GROUP_QUAD_VERTEX_SHADER, &SOLID_LAYER_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);
    let paint = |b: quanta::RenderBuilder| {
        draw_quad(
            b.clear(Color::rgba(0.0, 0.0, 0.0, 1.0)),
            &solid,
            &vb,
            None,
            (16, 16),
        )
    };

    let first = gpu.render_group((16, 16), Format::RGBA8, paint).unwrap();
    let first_handle = first.handle();
    drop(first); // returns to the pool
    let second = gpu.render_group((16, 16), Format::RGBA8, paint).unwrap();
    assert_eq!(
        second.handle(),
        first_handle,
        "same-shape checkout after drop must reuse the pooled intermediate"
    );
    // A DIFFERENT shape while `second` is out gets a fresh texture.
    let third = gpu.render_group((8, 8), Format::RGBA8, paint).unwrap();
    assert_ne!(third.handle(), second.handle());
}

#[test]
fn groups_nest_as_ordered_passes() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(
        &gpu,
        &[
            &GROUP_QUAD_VERTEX_SHADER,
            &SOLID_LAYER_FRAG_SHADER,
            &SAMPLE_LAYER_FRAG_SHADER,
        ],
    ) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let solid = pipeline(&gpu, &GROUP_QUAD_VERTEX_SHADER, &SOLID_LAYER_FRAG_SHADER);
    let sampling = pipeline(&gpu, &GROUP_QUAD_VERTEX_SHADER, &SAMPLE_LAYER_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    // Outer group samples the inner group — two prior passes, ordered.
    let outer = gpu
        .render_group((16, 16), Format::RGBA8, |b| {
            let inner = gpu.render_group((16, 16), Format::RGBA8, |ib| {
                draw_quad(
                    ib.clear(Color::rgba(0.0, 0.0, 0.0, 1.0)),
                    &solid,
                    &vb,
                    None,
                    (16, 16),
                )
            })?;
            draw_quad(
                b.clear(Color::rgba(0.0, 0.0, 0.0, 1.0)),
                &sampling,
                &vb,
                Some(&inner),
                (16, 16),
            )
        })
        .expect("nested groups render");

    let (w, h) = (4u32, 4u32);
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let mut pulse = gpu
        .render_into(&target, |b| {
            draw_quad(
                b.color_targets(vec![
                    ColorTarget::new(&target)
                        .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                        .with_store_op(StoreOp::Store),
                ]),
                &sampling,
                &vb,
                Some(&outer),
                (w, h),
            )
        })
        .unwrap();
    pulse.wait().unwrap();
    let px = target.read().unwrap();
    expect_rgb(&px, w, 1, 1, (51, 153, 255), "nested layer chain");
}

#[test]
fn msaa_group_resolves_into_the_pooled_layer() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&GROUP_QUAD_VERTEX_SHADER, &SOLID_LAYER_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    // The msaa(4) pass binds the 4-sample intermediate, so the
    // pipeline must be created 4-sample too (validate_pass_shape
    // rejects the mismatch loudly — by design).
    let layouts = pos_uv_layout();
    let solid = gpu
        .pipeline(
            &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
                vertex: &GROUP_QUAD_VERTEX_SHADER,
                fragment: &SOLID_LAYER_FRAG_SHADER,
            })
            .with_entries(
                GROUP_QUAD_VERTEX_SHADER.entry_point,
                SOLID_LAYER_FRAG_SHADER.entry_point,
            )
            .with_color_formats(vec![Format::RGBA8])
            .with_vertex_layouts(&layouts)
            .with_blend(quanta::BlendState::NONE)
            .with_sample_count(4),
        )
        .expect("4-sample pipeline");
    let vb = fullscreen_vb(&gpu);

    // `.msaa(4)` inside the group: the multisampled pass resolves into
    // the pooled single-sample layer, which reads back resolved.
    let layer = gpu
        .render_group((8, 8), Format::RGBA8, |b| {
            let mut pulse = draw_quad(
                b.msaa(4)
                    .msaa_resolve()
                    .clear(Color::rgba(0.0, 0.0, 0.0, 1.0)),
                &solid,
                &vb,
                None,
                (8, 8),
            )?;
            // The HOST reads this layer back below — the one case the
            // group contract requires a wait, and it happens here,
            // inside the closure (GPU-side consumers never need it).
            pulse.wait()?;
            Ok(pulse)
        })
        .expect("msaa group renders");
    let px = layer.read().expect("resolved layer reads back");
    expect_rgb(&px, 8, 4, 4, (51, 153, 255), "resolved msaa layer");
}
