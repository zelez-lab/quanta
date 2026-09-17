#![cfg(all(feature = "render", feature = "std"))]
//! Deferred render submission — render passes and resolves ride the
//! per-device pending lane.
//!
//! The contract under test: `RenderBuilder::pulse()` and
//! `resolve_texture` ENCODE (nothing reaches the queue until a sync
//! point), program order is preserved inside the one command buffer
//! (render-then-sample, compute→render), every documented sync point
//! completes the pending passes (`Pulse::wait`, `Texture::read`,
//! `SurfaceFrame::present`, `gpu.submit()`, the auto-submit
//! threshold), and resources dropped while the batch is still open
//! stay alive until it has executed. On a backend whose batches take
//! no render work the passes submit one by one and the same
//! assertions must still hold — the `__pending_encodes` probe is what
//! tells the two apart.

use quanta::RenderGpu;
use quanta::{Color, FieldUsage, Format};
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
struct DeferVary {
    #[position]
    clip: Vec4,
    uv: Vec2,
}

#[quanta::vertex]
fn defer_quad_vertex(pos: quanta::Vec3, uv: Vec2) -> DeferVary {
    DeferVary {
        clip: Vec4::new(pos.x, pos.y, 0.0, 1.0),
        uv,
    }
}

#[quanta::fragment]
fn defer_solid_frag() -> Vec4 {
    Vec4::new(0.2, 0.6, 1.0, 1.0)
}

#[quanta::fragment]
fn defer_sample_frag(s: DeferVary) -> Vec4 {
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

/// The solid fragment's color, as bytes.
const SOLID: (u8, u8, u8) = (51, 153, 255);

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

/// Whether this device coalesces passes in the lane: a clear-only
/// pass leaves one encode pending. `Some(false)` on a per-submission
/// backend (WebGPU) — the tests then only check the semantics;
/// `None` on a device with no render face at all (the CPU software
/// device) — the test skips.
fn batches_render(gpu: &quanta::Gpu) -> Option<bool> {
    let probe = gpu.render_target(2, 2, Format::RGBA8).ok()?;
    let builder = match gpu.render(&probe) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: no render path: {e}");
            return None;
        }
    };
    let _ = builder.clear(Color::BLACK).pulse().unwrap();
    let pending = gpu.__pending_encodes() > 0;
    gpu.flush().unwrap();
    Some(pending)
}

#[test]
fn passes_stay_pending_until_a_texture_read() {
    let Some(gpu) = try_gpu() else { return };
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    let a = gpu.render_target(4, 4, Format::RGBA8).unwrap();
    let b = gpu.render_target(4, 4, Format::RGBA8).unwrap();
    let _ = gpu
        .render(&a)
        .unwrap()
        .clear(Color::rgba(1.0, 0.0, 0.0, 1.0))
        .pulse()
        .unwrap();
    let _ = gpu
        .render(&b)
        .unwrap()
        .clear(Color::rgba(0.0, 1.0, 0.0, 1.0))
        .pulse()
        .unwrap();
    if batching {
        assert_eq!(
            gpu.__pending_encodes(),
            2,
            "two passes encoded, none submitted"
        );
    }
    // A host read of EITHER target completes the whole lane — the
    // pass into `a` included, though `a` is read second.
    let pb = b.read().unwrap();
    assert_eq!(gpu.__pending_encodes(), 0, "the read submitted the batch");
    expect_rgb(&pb, 4, 1, 1, (0, 255, 0), "second pass");
    let pa = a.read().unwrap();
    expect_rgb(&pa, 4, 1, 1, (255, 0, 0), "first pass");
}

#[test]
fn pulse_wait_and_submit_are_sync_points() {
    let Some(gpu) = try_gpu() else { return };
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    let t = gpu.render_target(4, 4, Format::RGBA8).unwrap();
    let mut pulse = gpu
        .render(&t)
        .unwrap()
        .clear(Color::rgba(0.0, 0.0, 1.0, 1.0))
        .pulse()
        .unwrap();
    if batching {
        assert_eq!(gpu.__pending_encodes(), 1);
    }
    pulse.wait().unwrap();
    assert_eq!(gpu.__pending_encodes(), 0, "wait flushed the lane");
    expect_rgb(&t.read().unwrap(), 4, 0, 0, (0, 0, 255), "waited pass");

    // submit(): the kick — submitted, not waited, still completed by
    // the next sync point.
    let _ = gpu
        .render(&t)
        .unwrap()
        .clear(Color::rgba(1.0, 1.0, 0.0, 1.0))
        .pulse()
        .unwrap();
    gpu.submit().unwrap();
    assert_eq!(gpu.__pending_encodes(), 0, "submit emptied the open batch");
    expect_rgb(&t.read().unwrap(), 4, 3, 3, (255, 255, 0), "kicked pass");
}

#[test]
fn resolve_is_deferred_and_a_read_completes_it() {
    let Some(gpu) = try_gpu() else { return };
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    let msaa = gpu.msaa_target(8, 8, Format::RGBA8, 4).unwrap();
    let dst = gpu.render_target(8, 8, Format::RGBA8).unwrap();
    let _ = gpu
        .render(&msaa)
        .unwrap()
        .clear(Color::rgba(0.0, 1.0, 1.0, 1.0))
        .pulse()
        .unwrap();
    match gpu.resolve_texture(&msaa, &dst) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("SKIP: resolve_texture not supported: {e}");
            return;
        }
    }
    if batching {
        assert_eq!(
            gpu.__pending_encodes(),
            2,
            "the resolve rides the lane behind the pass"
        );
    }
    let px = dst.read().unwrap();
    assert_eq!(gpu.__pending_encodes(), 0);
    expect_rgb(&px, 8, 4, 4, (0, 255, 255), "deferred resolve");
}

#[test]
fn group_then_sample_share_one_batch() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(
        &gpu,
        &[
            &DEFER_QUAD_VERTEX_SHADER,
            &DEFER_SOLID_FRAG_SHADER,
            &DEFER_SAMPLE_FRAG_SHADER,
        ],
    ) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    let solid = pipeline(&gpu, &DEFER_QUAD_VERTEX_SHADER, &DEFER_SOLID_FRAG_SHADER);
    let sampling = pipeline(&gpu, &DEFER_QUAD_VERTEX_SHADER, &DEFER_SAMPLE_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    let layer = gpu
        .render_group((16, 16), Format::RGBA8, |b| {
            draw_quad(b.clear(Color::BLACK), &solid, &vb, None, (16, 16))
        })
        .unwrap();
    let target = gpu.render_target(8, 8, Format::RGBA8).unwrap();
    let _ = draw_quad(
        gpu.render(&target).unwrap().clear(Color::BLACK),
        &sampling,
        &vb,
        Some(&layer),
        (8, 8),
    )
    .unwrap();
    if batching {
        assert_eq!(
            gpu.__pending_encodes(),
            2,
            "group pass + sampling pass in ONE open batch"
        );
    }
    let px = target.read().unwrap();
    expect_rgb(&px, 8, 2, 2, SOLID, "sampled group, same batch");
    expect_rgb(&px, 8, 7, 7, SOLID, "sampled group corner");
}

#[test]
fn acquire_group_is_a_pooled_layer_without_a_pass() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(
        &gpu,
        &[
            &DEFER_QUAD_VERTEX_SHADER,
            &DEFER_SOLID_FRAG_SHADER,
            &DEFER_SAMPLE_FRAG_SHADER,
        ],
    ) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let solid = pipeline(&gpu, &DEFER_QUAD_VERTEX_SHADER, &DEFER_SOLID_FRAG_SHADER);
    let sampling = pipeline(&gpu, &DEFER_QUAD_VERTEX_SHADER, &DEFER_SAMPLE_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    // Checked out bare, drawn into later, sampled later still.
    let layer = gpu.acquire_group((16, 16), Format::RGBA8).unwrap();
    let handle = layer.handle();
    let _ = draw_quad(
        gpu.render(&layer).unwrap().clear(Color::BLACK),
        &solid,
        &vb,
        None,
        (16, 16),
    )
    .unwrap();
    let target = gpu.render_target(8, 8, Format::RGBA8).unwrap();
    let _ = draw_quad(
        gpu.render(&target).unwrap().clear(Color::BLACK),
        &sampling,
        &vb,
        Some(&layer),
        (8, 8),
    )
    .unwrap();
    expect_rgb(&target.read().unwrap(), 8, 3, 3, SOLID, "acquired layer");
    // The bare checkout is the same pool as render_group's: dropping
    // it hands the texture back, and the next same-shape checkout
    // reuses it.
    drop(layer);
    let again = gpu.acquire_group((16, 16), Format::RGBA8).unwrap();
    assert_eq!(again.handle(), handle, "returned to the group pool");
}

#[cfg(feature = "compute")]
#[quanta::kernel]
fn write_quad(out: &mut [f32]) {
    // Six vertices of a fullscreen quad, pos.xyz + uv, 5 floats each:
    // the compute face producing what the render face consumes.
    // Corner per vertex is 0,1,2,0,2,3 — packed two bits each so the
    // kernel stays arithmetic (no constant table for the lowering to
    // trip on).
    let i = quark_id();
    let v = i / 5;
    let c = i - v * 5;
    let k = (3620u32 >> (2 * v)) & 3;
    let right = ((k + 1) >> 1) & 1;
    let top = k >> 1;
    let x = (right as f32) * 2.0 - 1.0;
    let y = (top as f32) * 2.0 - 1.0;
    let u = right as f32;
    let w = top as f32;
    let m0 = if c == 0 { 1.0 } else { 0.0 };
    let m1 = if c == 1 { 1.0 } else { 0.0 };
    let m3 = if c == 3 { 1.0 } else { 0.0 };
    let m4 = if c == 4 { 1.0 } else { 0.0 };
    out[i] = x * m0 + y * m1 + u * m3 + w * m4;
}

#[cfg(feature = "compute")]
#[test]
fn compute_written_vertices_draw_in_the_same_batch() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&DEFER_QUAD_VERTEX_SHADER, &DEFER_SOLID_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let Ok(mut wave) = write_quad(&gpu) else {
        eprintln!("SKIP: no compute kernel binary");
        return;
    };
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    let solid = pipeline(&gpu, &DEFER_QUAD_VERTEX_SHADER, &DEFER_SOLID_FRAG_SHADER);
    // Storage for the kernel AND vertex data for the draw.
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(
            FULLSCREEN_QUAD.len(),
            FieldUsage::default_render().union(FieldUsage::default_compute()),
        )
        .unwrap();
    vb.write(&[0.0f32; 30]).unwrap();
    wave.bind(0, &vb);
    // Deferred dispatch, then a pass that vertex-pulls its output:
    // the in-buffer compute→render edge.
    let _ = gpu.dispatch(&wave, 30).unwrap();
    let target = gpu.render_target(8, 8, Format::RGBA8).unwrap();
    let _ = draw_quad(
        gpu.render(&target).unwrap().clear(Color::BLACK),
        &solid,
        &vb,
        None,
        (8, 8),
    )
    .unwrap();
    if batching {
        assert!(
            gpu.__pending_encodes() >= 1,
            "dispatch and pass share the open batch"
        );
    }
    let px = target.read().unwrap();
    expect_rgb(&px, 8, 1, 1, SOLID, "quad from compute-written vertices");
    expect_rgb(&px, 8, 6, 6, SOLID, "quad corner");
}

#[test]
fn resources_dropped_with_the_batch_open_stay_alive() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(
        &gpu,
        &[
            &DEFER_QUAD_VERTEX_SHADER,
            &DEFER_SOLID_FRAG_SHADER,
            &DEFER_SAMPLE_FRAG_SHADER,
        ],
    ) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let vb = fullscreen_vb(&gpu);
    let target = gpu.render_target(8, 8, Format::RGBA8).unwrap();
    {
        // Everything the two passes reference — the intermediate, both
        // pipelines, the vertex buffer's twin — dies before anything is
        // submitted. Metal retains through the encoders; Vulkan parks
        // the destroys behind the batch pins. Either way the read below
        // must see the drawn result, not freed memory (VVL on CI).
        let solid = pipeline(&gpu, &DEFER_QUAD_VERTEX_SHADER, &DEFER_SOLID_FRAG_SHADER);
        let sampling = pipeline(&gpu, &DEFER_QUAD_VERTEX_SHADER, &DEFER_SAMPLE_FRAG_SHADER);
        let mid = gpu.render_target(16, 16, Format::RGBA8).unwrap();
        let vb2 = fullscreen_vb(&gpu);
        let _ = draw_quad(
            gpu.render(&mid).unwrap().clear(Color::BLACK),
            &solid,
            &vb2,
            None,
            (16, 16),
        )
        .unwrap();
        let _ = draw_quad(
            gpu.render(&target).unwrap().clear(Color::BLACK),
            &sampling,
            &vb,
            Some(&mid),
            (8, 8),
        )
        .unwrap();
        drop(mid);
        drop(solid);
        drop(sampling);
        drop(vb2);
    }
    let px = target.read().unwrap();
    expect_rgb(&px, 8, 4, 4, SOLID, "drawn through dropped resources");
}

#[test]
fn present_submits_the_frame() {
    let Some(gpu) = try_gpu() else { return };
    if !gpu.supports_surface_present() {
        eprintln!("SKIP: no present path");
        return;
    }
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    let config = quanta::SurfaceConfig::new(32, 32);
    let mut surface = gpu
        .create_surface(&quanta::SurfaceTarget::Headless, &config)
        .unwrap();
    for _ in 0..3 {
        let frame = surface.acquire().unwrap();
        let _ = gpu
            .render(frame.texture())
            .unwrap()
            .clear(Color::rgba(0.5, 0.5, 0.5, 1.0))
            .pulse()
            .unwrap();
        let _ = gpu
            .render(frame.texture())
            .unwrap()
            .clear(Color::rgba(0.0, 0.0, 0.0, 1.0))
            .pulse()
            .unwrap();
        if batching {
            assert_eq!(gpu.__pending_encodes(), 2, "frame passes pending");
        }
        frame.present().unwrap();
        assert_eq!(gpu.__pending_encodes(), 0, "present submitted the frame");
    }
}

#[test]
fn threshold_submits_mid_frame_and_order_holds() {
    let Some(gpu) = try_gpu() else { return };
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    let t = gpu.render_target(4, 4, Format::RGBA8).unwrap();
    // 600 clear-only passes over one target: well past the 512-encode
    // threshold, so the lane submits at least once mid-stream. The
    // LAST clear must win — cross-batch ordering is queue order.
    for i in 0..600u32 {
        let v = (i % 7) as f32 / 7.0;
        let _ = gpu
            .render(&t)
            .unwrap()
            .clear(Color::rgba(v, 0.0, 1.0 - v, 1.0))
            .pulse()
            .unwrap();
    }
    if batching {
        assert!(
            gpu.__pending_encodes() < 600,
            "threshold submitted mid-stream"
        );
    }
    // i = 599 → 599 % 7 = 4 → r = 4/7.
    let px = t.read().unwrap();
    let want = (
        (4.0f32 / 7.0 * 255.0).round() as u8,
        0,
        ((3.0f32 / 7.0) * 255.0).round() as u8,
    );
    expect_rgb(&px, 4, 2, 2, want, "last of 600 passes");
}

#[test]
fn msaa_builder_pass_is_one_encode() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&DEFER_QUAD_VERTEX_SHADER, &DEFER_SOLID_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    let layouts = pos_uv_layout();
    let solid4 = gpu
        .pipeline(
            &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
                vertex: &DEFER_QUAD_VERTEX_SHADER,
                fragment: &DEFER_SOLID_FRAG_SHADER,
            })
            .with_entries(
                DEFER_QUAD_VERTEX_SHADER.entry_point,
                DEFER_SOLID_FRAG_SHADER.entry_point,
            )
            .with_color_formats(vec![Format::RGBA8])
            .with_vertex_layouts(&layouts)
            .with_blend(quanta::BlendState::NONE)
            .with_sample_count(4),
        )
        .expect("4-sample pipeline");
    let vb = fullscreen_vb(&gpu);
    let target = gpu.render_target(8, 8, Format::RGBA8).unwrap();
    let _ = draw_quad(
        gpu.render(&target)
            .unwrap()
            .msaa(4)
            .msaa_resolve()
            .clear(Color::BLACK),
        &solid4,
        &vb,
        None,
        (8, 8),
    )
    .unwrap();
    if batching {
        assert_eq!(gpu.__pending_encodes(), 1, "subpass resolve = one encode");
    }
    expect_rgb(
        &target.read().unwrap(),
        8,
        4,
        4,
        SOLID,
        "resolved msaa pass",
    );
}

#[test]
fn waiting_a_frame_pulse_leaves_later_frames_in_flight() {
    let Some(gpu) = try_gpu() else { return };
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    if !batching {
        eprintln!("SKIP: per-submission backend");
        return;
    }
    // Three "frames", each submitted with the kick (present's role
    // in a windowed loop), the third left open.
    let a = gpu.render_target(4, 4, Format::RGBA8).unwrap();
    let b = gpu.render_target(4, 4, Format::RGBA8).unwrap();
    let c = gpu.render_target(4, 4, Format::RGBA8).unwrap();
    let mut pulse_a = gpu
        .render(&a)
        .unwrap()
        .clear(Color::rgba(1.0, 0.0, 0.0, 1.0))
        .pulse()
        .unwrap();
    gpu.submit().unwrap();
    let mut pulse_b = gpu
        .render(&b)
        .unwrap()
        .clear(Color::rgba(0.0, 1.0, 0.0, 1.0))
        .pulse()
        .unwrap();
    gpu.submit().unwrap();
    let mut pulse_c = gpu
        .render(&c)
        .unwrap()
        .clear(Color::rgba(0.0, 0.0, 1.0, 1.0))
        .pulse()
        .unwrap();
    assert_eq!(gpu.__outstanding_batches(), 2, "two frames submitted");
    assert_eq!(gpu.__pending_encodes(), 1, "third frame still open");

    // The depth-N pattern: waiting the OLDEST frame must not drain
    // the newer ones — frame b stays in flight, frame c stays open.
    pulse_a.wait().unwrap();
    assert_eq!(gpu.__outstanding_batches(), 1, "frame b still in flight");
    assert_eq!(gpu.__pending_encodes(), 1, "frame c still open");
    expect_rgb(&a.read().unwrap(), 4, 0, 0, (255, 0, 0), "frame a");
    // (the read of `a` touched nothing newer either)
    assert_eq!(gpu.__outstanding_batches(), 1);
    assert_eq!(gpu.__pending_encodes(), 1);

    // Waiting the open frame's pulse submits it and completes through
    // it — which includes the older frame b (queue order).
    pulse_c.wait().unwrap();
    assert_eq!(gpu.__outstanding_batches(), 0);
    assert_eq!(gpu.__pending_encodes(), 0);
    pulse_b.wait().unwrap(); // already complete: a no-op
    expect_rgb(&b.read().unwrap(), 4, 0, 0, (0, 255, 0), "frame b");
    expect_rgb(&c.read().unwrap(), 4, 0, 0, (0, 0, 255), "frame c");
}

#[test]
fn texture_byte_ops_wait_only_the_batches_that_touched_them() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(
        &gpu,
        &[&DEFER_QUAD_VERTEX_SHADER, &DEFER_SAMPLE_FRAG_SHADER],
    ) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let Some(batching) = batches_render(&gpu) else {
        return;
    };
    if !batching {
        eprintln!("SKIP: per-submission backend");
        return;
    }
    let sampling = pipeline(&gpu, &DEFER_QUAD_VERTEX_SHADER, &DEFER_SAMPLE_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);
    // The atlas shape: an uploadable texture sampled by frame 1,
    // presented (submitted), then re-uploaded while frame 2 — which
    // never touches it — is being encoded. The upload must wait frame
    // 1 only, and frame 1 must have sampled the OLD contents.
    let atlas = gpu.texture(4, 4).unwrap();
    atlas.write(&[255u8; 4 * 4 * 4]).unwrap();
    let sampled = gpu.render_target(4, 4, Format::RGBA8).unwrap();
    let other = gpu.render_target(4, 4, Format::RGBA8).unwrap();
    let _ = draw_quad(
        gpu.render(&sampled).unwrap().clear(Color::BLACK),
        &sampling,
        &vb,
        Some(&atlas),
        (4, 4),
    )
    .unwrap();
    gpu.submit().unwrap(); // frame 1 submitted — it sampled `atlas`
    let _ = gpu
        .render(&other)
        .unwrap()
        .clear(Color::rgba(0.0, 0.0, 0.0, 1.0))
        .pulse()
        .unwrap();
    gpu.submit().unwrap(); // frame 2 submitted, never touched `atlas`
    let _ = gpu
        .render(&other)
        .unwrap()
        .clear(Color::rgba(0.5, 0.5, 0.5, 1.0))
        .pulse()
        .unwrap(); // frame 3 open, never touched `atlas`
    assert_eq!(gpu.__outstanding_batches(), 2);
    assert_eq!(gpu.__pending_encodes(), 1);

    atlas.write(&[7u8; 4 * 4 * 4]).unwrap();
    assert_eq!(
        gpu.__outstanding_batches(),
        1,
        "the upload waited frame 1 (which sampled the atlas) and nothing newer"
    );
    assert_eq!(
        gpu.__pending_encodes(),
        1,
        "the open frame was not submitted"
    );
    // Frame 1 ran BEFORE the upload landed: it sampled white, not 7s.
    expect_rgb(
        &sampled.read().unwrap(),
        4,
        1,
        1,
        (255, 255, 255),
        "frame 1 sampled the old atlas",
    );
    assert_eq!(
        gpu.__pending_encodes(),
        1,
        "reading a completed target submits nothing"
    );

    // A read of a texture the OPEN frame draws into submits and
    // completes through it (queue order carries frame 2 along).
    let px = other.read().unwrap();
    assert_eq!(gpu.__outstanding_batches(), 0);
    assert_eq!(gpu.__pending_encodes(), 0);
    expect_rgb(
        &px,
        4,
        1,
        1,
        (128, 128, 128),
        "open frame completed by the read",
    );
    expect_rgb(
        &atlas.read().unwrap(),
        4,
        1,
        1,
        (7, 7, 7),
        "upload landed after frame 1",
    );
}
