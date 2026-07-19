#![cfg(feature = "render")]
//! Render-resource lifecycle — proves the leak-by-design is gone.
//!
//! Every `Texture` / `Sampler` / `Pipeline` / `TextureView` /
//! `OcclusionQuery` wrapper now releases its driver registry entry on
//! Drop (`device + live` pattern). These tests snapshot the driver's
//! registry sizes around create+drop cycles and assert the entries are
//! freed, plus a 100+ frame create/drop loop asserting the registries
//! do not grow unboundedly — the test that would have caught the leak.
//!
//! Requires a GPU; skips gracefully if none available.

use quanta::RenderGpu;

use quanta::{Color, Format, SamplerDesc, TextureDesc, TextureUsage, TextureViewDesc};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// ─── Shaders (pipeline lifecycle) ───────────────────────────────────────────

#[quanta::vertex]
fn lifecycle_vertex(pos: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn lifecycle_fragment() -> Vec4 {
    Vec4::new(0.0, 0.0, 1.0, 1.0)
}

fn have_shaders(gpu: &quanta::Gpu) -> bool {
    // The driver picks the payload from the binaries; skip when this
    // vendor has none compiled in.
    LIFECYCLE_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_some()
        && LIFECYCLE_FRAGMENT_SHADER
            .for_vendor(gpu.caps().vendor)
            .is_some()
}

fn vertex_layout() -> Vec<quanta::VertexLayout> {
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

fn pipeline_desc<'a>(layouts: &'a [quanta::VertexLayout]) -> quanta::PipelineDesc<'a> {
    quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
        vertex: &LIFECYCLE_VERTEX_SHADER,
        fragment: &LIFECYCLE_FRAGMENT_SHADER,
    })
    .with_entries(
        LIFECYCLE_VERTEX_SHADER.entry_point,
        LIFECYCLE_FRAGMENT_SHADER.entry_point,
    )
    .with_vertex_layouts(layouts)
    .with_color_formats(vec![Format::RGBA8])
    .with_blend(quanta::BlendState::NONE)
}

fn small_texture_desc() -> TextureDesc {
    TextureDesc::new(16, 16, Format::RGBA8)
        .with_usage(TextureUsage::SHADER_READ.union(TextureUsage::STORAGE))
}

// ─── Single-resource lifecycle ──────────────────────────────────────────────

#[test]
fn texture_drop_frees_registry_entry() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let before = gpu.debug_registry_counts();
    let tex = gpu.create_texture(&small_texture_desc()).unwrap();
    let during = gpu.debug_registry_counts();
    assert_ne!(
        before, during,
        "texture_create should register a driver-side entry"
    );
    drop(tex);
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "dropping a Texture must free its entry");
}

#[cfg(feature = "software")]
#[test]
fn cpu_texture_drop_frees_registry_entry() {
    let gpu = quanta::init_cpu();
    let before = gpu.debug_registry_counts();
    let tex = gpu.create_texture(&small_texture_desc()).unwrap();
    let during = gpu.debug_registry_counts();
    assert_ne!(
        before, during,
        "CPU texture_create should register a buffer entry"
    );
    drop(tex);
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "dropping a CPU Texture must free its entry");
}

#[test]
fn sampler_drop_frees_registry_entry() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let before = gpu.debug_registry_counts();
    let sampler = gpu.sampler(&SamplerDesc::default()).unwrap();
    let during = gpu.debug_registry_counts();
    assert_ne!(
        before, during,
        "sampler_create should register a driver-side entry"
    );
    drop(sampler);
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "dropping a Sampler must free its entry");
}

#[test]
fn pipeline_drop_frees_registry_entry() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !have_shaders(&gpu) {
        eprintln!("skipping: no shader for this vendor");
        return;
    }

    let layouts = vertex_layout();
    let before = gpu.debug_registry_counts();
    let pipeline = gpu.pipeline(&pipeline_desc(&layouts)).unwrap();
    let during = gpu.debug_registry_counts();
    assert_ne!(
        before, during,
        "pipeline_create should register a driver-side entry"
    );
    drop(pipeline);
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "dropping a Pipeline must free its entry");
}

#[test]
fn texture_view_drop_frees_registry_entry() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(&small_texture_desc().with_mip_levels(4))
        .unwrap();

    let with_texture = gpu.debug_registry_counts();
    let view = match gpu.texture_view_create(
        &tex,
        &TextureViewDesc {
            format: None,
            mip_range: 0..4,
            layer_range: 0..1,
        },
    ) {
        Ok(view) => view,
        Err(e) => {
            eprintln!("texture views not supported: {}", e);
            return;
        }
    };
    let with_view = gpu.debug_registry_counts();
    assert_ne!(
        with_texture, with_view,
        "texture_view_create should register a driver-side entry"
    );
    drop(view);
    let after_view = gpu.debug_registry_counts();
    assert_eq!(
        with_texture, after_view,
        "dropping a TextureView must free its entry"
    );
}

#[test]
fn occlusion_query_drop_frees_registry_entry() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let before = gpu.debug_registry_counts();
    let query = match gpu.occlusion_query_create(4) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("occlusion queries not supported: {}", e);
            return;
        }
    };
    let during = gpu.debug_registry_counts();
    assert_ne!(
        before, during,
        "occlusion_query_create should register a driver-side entry"
    );
    drop(query);
    let after = gpu.debug_registry_counts();
    assert_eq!(
        before, after,
        "dropping an OcclusionQuery must free its entry"
    );
}

// ─── Double-free safety ─────────────────────────────────────────────────────

#[test]
fn explicit_view_destroy_does_not_double_free() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(&small_texture_desc().with_mip_levels(2))
        .unwrap();
    let with_texture = gpu.debug_registry_counts();

    let view = match gpu.texture_view_create(
        &tex,
        &TextureViewDesc {
            format: None,
            mip_range: 0..2,
            layer_range: 0..1,
        },
    ) {
        Ok(view) => view,
        Err(e) => {
            eprintln!("texture views not supported: {}", e);
            return;
        }
    };

    // Explicit destroy consumes the view and disarms its Drop — the
    // handle must be destroyed exactly once.
    gpu.texture_view_destroy(view).unwrap();
    let after = gpu.debug_registry_counts();
    assert_eq!(
        with_texture, after,
        "explicit texture_view_destroy must free exactly one entry"
    );
}

fn pass_through(tex: quanta::Texture) -> quanta::Texture {
    tex
}

#[test]
fn moved_texture_frees_exactly_once() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let before = gpu.debug_registry_counts();

    // Move the texture through a function and a Vec: only the final
    // owner's Drop may release the handle.
    let tex = gpu.create_texture(&small_texture_desc()).unwrap();
    let tex = pass_through(tex);
    let mut owners = vec![tex];
    let tex = owners.pop().unwrap();
    drop(owners);

    let during = gpu.debug_registry_counts();
    assert_ne!(before, during, "moves must not release the entry early");

    drop(tex);
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "moved Texture must be freed exactly once");

    // The device must still be healthy afterwards (no over-release).
    let probe = gpu.create_texture(&small_texture_desc()).unwrap();
    probe.write(&vec![7u8; 16 * 16 * 4]).unwrap();
    assert_eq!(probe.read().unwrap()[0], 7);
}

// ─── 100-frame reuse loop (the test that would have caught the leak) ───────

#[test]
fn hundred_frame_reuse_does_not_grow_registries() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let before = gpu.debug_registry_counts();
    let shaders = have_shaders(&gpu);
    let layouts = vertex_layout();

    for _frame in 0..120 {
        // Per-frame resources, all dropped at the end of the frame.
        let target = gpu
            .render_target(64, 64, Format::RGBA8)
            .expect("render target");
        let _sampler = gpu.sampler(&SamplerDesc::default()).expect("sampler");
        let _pipeline = shaders.then(|| gpu.pipeline(&pipeline_desc(&layouts)).expect("pipeline"));

        // Render a real frame into the target so the resources are used.
        let mut pulse = gpu
            .render(&target)
            .expect("render pass")
            .clear(Color::BLACK)
            .pulse()
            .expect("submit");
        pulse.wait().expect("wait");
    }

    let after = gpu.debug_registry_counts();
    assert_eq!(
        before, after,
        "120 create+drop frames must not grow the driver registries"
    );
}
