#![cfg(feature = "render")]
//! `PipelineDesc::color_formats` is **per-attachment** — entry `i` types
//! color attachment `i` of the pass — and this suite proves the two
//! checks that enforce it.
//!
//! The bug that motivated them: a consumer read `color_formats` as "the
//! formats this pipeline may be used against", declared `[BGRA8, RGBA8]`
//! for a single `RGBA8` target, and got a phantom attachment 1 plus a
//! mis-typed attachment 0 — which Metal accepted silently, then dropped
//! draws for. Two layers close it:
//!
//!  * **Creation-time (K3)**: a descriptor that declares more color
//!    attachments than the fragment writes is rejected by
//!    `gpu.pipeline()` with `CompilationFailed`, reflected from the
//!    SPIR-V fragment. The DSL always emits single-output fragments, so
//!    a two-format desc over any DSL fragment is the easy repro. Runs on
//!    the CPU backend (the check is at the API layer, before the driver)
//!    and only when a SPIR-V payload is embedded (needs `QUANTA_COMPILER`
//!    at build time; skips otherwise).
//!
//!  * **Encode-time (K2)**: at `pulse()`, the bound color/depth targets
//!    are checked against the pipeline's declared formats — count first,
//!    then per-attachment format, then depth presence — with a named
//!    `InvalidParam`. Always-on and backend-agnostic. Needs a live GPU
//!    (the CPU backend has no render encoder); skips without one.
//!
//! `N < M` (writing fewer attachments than the fragment declares) is the
//! driver-legal partial case and is **allowed** — see the `spirv_meta`
//! unit tests for the reflection polarity, and the fact that the whole
//! render suite (single-output fragments, single-format pipelines) stays
//! green as the proof that neither check misfires on the legitimate
//! shapes.
//!
//! Run (Metal): cargo test --test color_format_validation --features metal,render -- --test-threads=1
//! Run (CPU-only K3): cargo test --test color_format_validation --features software

use quanta::RenderGpu;

use quanta::render_pass::ColorTarget;
use quanta::{Format, PipelineDesc, QuantaErrorKind, ShaderSource};

// A single-output fragment (every DSL fragment is single-output) and a
// trivial vertex, shared by the tests below.
#[quanta::vertex]
fn cfv_vertex(pos: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn cfv_fragment() -> Vec4 {
    Vec4::new(1.0, 0.0, 0.0, 1.0)
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

fn shaders() -> ShaderSource<'static> {
    ShaderSource::Binaries {
        vertex: &CFV_VERTEX_SHADER,
        fragment: &CFV_FRAGMENT_SHADER,
    }
}

fn desc_with_formats(
    layouts: &[quanta::VertexLayout],
    color_formats: Vec<Format>,
) -> PipelineDesc<'_> {
    PipelineDesc::new(shaders())
        .with_entries(
            CFV_VERTEX_SHADER.entry_point,
            CFV_FRAGMENT_SHADER.entry_point,
        )
        .with_vertex_layouts(layouts)
        .with_color_formats(color_formats)
}

// ─── K3: creation-time fragment-output reflection ────────────────────────────
//
// The reflection check lives at the API layer, before any driver call,
// so any backend exercises it — these use a live GPU via `try_gpu()`
// (rejection happens before the driver sees the descriptor). They
// require the fragment's SPIR-V to be embedded; if the build had no
// `QUANTA_COMPILER`, `spirv` is `None` and the check correctly skips, so
// the test skips too.

fn fragment_has_spirv() -> bool {
    CFV_FRAGMENT_SHADER.spirv.is_some()
}

/// A descriptor declaring TWO color attachments over a single-output
/// fragment is rejected at creation, naming both counts.
#[test]
fn n_greater_than_m_rejected_at_creation() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if !fragment_has_spirv() {
        return; // no SPIR-V payload embedded (compiler-free build) — check skips.
    }
    let layouts = vertex_layout();
    let desc = desc_with_formats(&layouts, vec![Format::RGBA8, Format::RGBA8]);
    match gpu.pipeline(&desc) {
        Err(e) => {
            assert!(
                matches!(e.kind, QuantaErrorKind::CompilationFailed(_)),
                "expected CompilationFailed, got {:?}",
                e.kind
            );
            let msg = format!("{e:?}");
            assert!(
                msg.contains("declares 2 color attachments") && msg.contains("fragment writes 1"),
                "error should name both counts, got: {msg}"
            );
        }
        Ok(_) => panic!("2-format desc over a single-output fragment must be rejected at creation"),
    }
}

/// The matched case (one format, one fragment output) builds fine —
/// proof the check's polarity is right (it rejects only N > M).
#[test]
fn n_equal_m_allowed_at_creation() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if !fragment_has_spirv() {
        return;
    }
    let layouts = vertex_layout();
    let desc = desc_with_formats(&layouts, vec![Format::RGBA8]);
    assert!(
        gpu.pipeline(&desc).is_ok(),
        "a single-format desc over a single-output fragment must build"
    );
}

/// `N < M` is allowed: `__check_fragment_outputs` (the shared reflection
/// entry point) accepts declaring fewer attachments than the fragment
/// writes. Proven here with a hand-assembled two-output fragment module,
/// so the verdict does not depend on the DSL (which only emits one
/// output) or on a live GPU. This is the direct, backend-free evidence
/// for the N<M contract.
#[test]
fn n_less_than_m_allowed_by_reflection() {
    // Minimal SPIR-V (v1.3) fragment module with two Location-decorated
    // Output variables in the entry interface: ids %10 (Location 0) and
    // %11 (Location 1).
    let name_main = u32::from_le_bytes([b'm', b'a', b'i', b'n']);
    #[rustfmt::skip]
    let words: Vec<u32> = vec![
        0x0723_0203, 0x0001_0300, 0, 100, 0,   // header
        (2u32 << 16) | 17, 1,                   // OpCapability Shader
        // OpEntryPoint Fragment %1 "main" %10 %11  (7 words)
        (7u32 << 16) | 15, 4, 1, name_main, 0, 10, 11,
        (4u32 << 16) | 71, 10, 30, 0,           // OpDecorate %10 Location 0
        (4u32 << 16) | 71, 11, 30, 1,           // OpDecorate %11 Location 1
        (4u32 << 16) | 59, 2, 10, 3,            // OpVariable %2 %10 Output
        (4u32 << 16) | 59, 2, 11, 3,            // OpVariable %2 %11 Output
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    // N < M (1 declared, 2 written) — allowed.
    assert!(
        quanta::__check_fragment_outputs(Some(&bytes), 1).is_ok(),
        "N < M must be allowed (partial write is driver-legal)"
    );
    // N == M — allowed.
    assert!(quanta::__check_fragment_outputs(Some(&bytes), 2).is_ok());
    // N > M — rejected.
    assert!(
        quanta::__check_fragment_outputs(Some(&bytes), 3).is_err(),
        "N > M must be rejected"
    );
}

// ─── K2: encode-time pass-shape validation ───────────────────────────────────
//
// These need a live render encoder, so they run on a real GPU and skip
// gracefully without one. Each pipeline is created with a shape that
// PASSES the creation check (so we isolate the encode-time check), then
// bound against a deliberately mismatched pass.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// A pipeline declaring ONE color attachment, bound against a pass with
/// TWO color targets, fails at `pulse()` naming both counts. (The
/// two-format-over-one-target direction is caught earlier, at creation,
/// by K3 — this is the mismatch that only surfaces at encode time.)
#[test]
fn count_mismatch_named_err_at_encode() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    // One declared color attachment — passes the creation check.
    let pipeline = gpu
        .pipeline(&desc_with_formats(&layouts, vec![Format::RGBA8]))
        .expect("single-format pipeline should build");

    let t0 = gpu.render_target(32, 32, Format::RGBA8).unwrap();
    let t1 = gpu.render_target(32, 32, Format::RGBA8).unwrap();

    let verts: [f32; 9] = [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), quanta::FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    let result = gpu
        .render(&t0)
        .unwrap()
        .color_targets(vec![ColorTarget::new(&t0), ColorTarget::new(&t1)])
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse();

    match result {
        Err(e) => {
            assert!(
                matches!(e.kind, QuantaErrorKind::InvalidParam(_)),
                "expected InvalidParam, got {:?}",
                e.kind
            );
            let msg = format!("{e:?}");
            assert!(
                msg.contains("declares 1 color attachments")
                    && msg.contains("binds 2 color targets"),
                "error should name both counts, got: {msg}"
            );
        }
        Ok(_) => panic!("1-attachment pipeline over a 2-target pass must fail at pulse()"),
    }
}

/// A pipeline whose `color_formats[0]` is `BGRA8`, bound against a single
/// `RGBA8` target, fails at `pulse()` naming the attachment index, the
/// expected format and the bound format. This is the downstream ask's
/// "optional bind-time check", generalized to every attachment.
#[test]
fn format_mismatch_named_err_at_encode() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    // Declared BGRA8 (one attachment — passes the creation check).
    let pipeline = gpu
        .pipeline(&desc_with_formats(&layouts, vec![Format::BGRA8]))
        .expect("single-format pipeline should build");

    // ...but the bound target is RGBA8.
    let target = gpu.render_target(32, 32, Format::RGBA8).unwrap();

    let verts: [f32; 9] = [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), quanta::FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    let result = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![ColorTarget::new(&target)])
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse();

    match result {
        Err(e) => {
            assert!(
                matches!(e.kind, QuantaErrorKind::InvalidParam(_)),
                "expected InvalidParam, got {:?}",
                e.kind
            );
            let msg = format!("{e:?}");
            assert!(
                msg.contains("color target 0 format mismatch")
                    && msg.contains("BGRA8")
                    && msg.contains("RGBA8"),
                "error should name index, expected and got, got: {msg}"
            );
        }
        Ok(_) => panic!("BGRA8 pipeline over an RGBA8 target must fail at pulse()"),
    }
}

/// A correctly-shaped pass (one BGRA8 attachment, one BGRA8 target) still
/// draws — the encode-time check does not misfire on the matched case.
#[test]
fn matched_shape_still_draws() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let pipeline = gpu
        .pipeline(&desc_with_formats(&layouts, vec![Format::BGRA8]))
        .expect("pipeline should build");

    let target = gpu.render_target(32, 32, Format::BGRA8).unwrap();

    let verts: [f32; 9] = [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), quanta::FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![ColorTarget::new(&target)])
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .expect("a matched color-format pass must not be rejected");
    pulse.wait().unwrap();
}

/// The legacy single-target path (no explicit `color_targets` — the
/// primary render target is the sole color attachment) with a matched
/// single-format pipeline still draws. This exercises the
/// `color_targets.is_empty()` branch of the shape check, where the
/// bound count is the one implicit attachment and its format is the
/// primary target's.
#[test]
fn legacy_single_target_pass_still_draws() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    // One RGBA8 attachment, matching the primary target below.
    let pipeline = gpu
        .pipeline(&desc_with_formats(&layouts, vec![Format::RGBA8]))
        .expect("pipeline should build");

    let target = gpu.render_target(32, 32, Format::RGBA8).unwrap();

    let verts: [f32; 9] = [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), quanta::FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    // No `.color_targets(...)` — the implicit single-target path.
    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .expect("a matched single-target pass must not be rejected");
    pulse.wait().unwrap();
}
