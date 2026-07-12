//! Integration tests: full vertex/fragment MSL emission + metallib compile.
//!
//! The unit tests in `shader_ast_tests.rs` assert on emitted MSL text; these
//! close the loop by feeding the emitted MSL to `xcrun metal`/`metallib` (the
//! same path the shader pipeline uses) and asserting the driver accepts it.
//! They gate on `xcrun` presence like the existing metallib tests, so they
//! `SKIP` on non-Apple hosts / hosts without the Metal toolchain.
//!
//! The three fixtures are the dija render shaders (rect / glyph / image) —
//! copied from `dija-render/src/quanta_backend/{shaders.rs,shaders_textured.rs}`.
//! Each had NEVER produced a working metallib before this emitter (the
//! consumer fell back to hand-written MSL); these lock in that they now do.
//!
//! Fixture bodies are written in clean Rust source. `syn` parses that
//! identically to the token-stringified form the proc macro actually ships
//! (verified: `Vec4 :: new` and `Vec4::new` parse to the same AST), so the
//! fixtures stay readable while exercising the real emit path. Texture fragment
//! bodies use the canonical `sample(N, uv)` slot form the macro's
//! `rewrite_texture_names` produces before the wire — that's what the compiler
//! receives.

use quanta_ir::{ShaderDef, ShaderParam, ShaderStage, ShaderType};

use super::{emit_fragment_shader, emit_vertex_shader};
use crate::metallib::compile_msl_to_metallib;

fn xcrun_present() -> bool {
    std::process::Command::new("xcrun")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn p(name: &str, ty: ShaderType) -> ShaderParam {
    ShaderParam {
        name: name.to_string(),
        ty,
        is_uniform: false,
    }
}

fn uniform(name: &str, ty: ShaderType) -> ShaderParam {
    ShaderParam {
        name: name.to_string(),
        ty,
        is_uniform: true,
    }
}

/// Emit MSL, then compile it to metallib; assert the driver accepts it.
/// Returns the emitted MSL for extra assertions.
fn must_compile(msl: &str, label: &str) {
    match compile_msl_to_metallib(msl) {
        Ok(Some(bytes)) => assert!(!bytes.is_empty(), "{label}: empty metallib"),
        Ok(None) => panic!("{label}: xcrun unexpectedly absent after presence check"),
        Err(e) => panic!("{label}: metallib compile failed:\n{e}\n--- MSL ---\n{msl}"),
    }
}

use ShaderType::{F32, Vec2, Vec4};

// ─── rect ─────────────────────────────────────────────────────────────────

fn rect_vertex_def() -> ShaderDef {
    ShaderDef {
        name: "rect_vertex".into(),
        stage: ShaderStage::Vertex,
        params: vec![
            p("pos", Vec2),
            p("corner", Vec2),
            p("size", Vec2),
            p("color", Vec4),
            p("border_color", Vec4),
            p("corner_radii", Vec4),
            p("border_width", F32),
            p("shape_type", F32),
            uniform("viewport", Vec2),
        ],
        return_type: Vec4,
        body_source: r#"{
            let px = pos.x + corner.x * size.x;
            let py = pos.y + corner.y * size.y;
            let ndc_x = px / viewport.x * 2.0 - 1.0;
            let ndc_y = 1.0 - py / viewport.y * 2.0;
            Vec4::new(ndc_x, ndc_y, 0.0, 1.0)
        }"#
        .into(),
    }
}

fn rect_fragment_def() -> ShaderDef {
    ShaderDef {
        name: "rect_fragment".into(),
        stage: ShaderStage::Fragment,
        params: vec![
            p("corner", Vec2),
            p("size", Vec2),
            p("color", Vec4),
            p("border_color", Vec4),
            p("corner_radii", Vec4),
            p("border_width", F32),
            p("shape_type", F32),
        ],
        return_type: Vec4,
        body_source: r#"{
            let px = corner.x * size.x;
            let py = corner.y * size.y;
            let half_x = size.x * 0.5;
            let half_y = size.y * 0.5;
            let cpx = px - half_x;
            let cpy = py - half_y;
            let dist = if shape_type > 0.5 {
                let nx = cpx / half_x;
                let ny = cpy / half_y;
                let d = length(Vec2::new(nx, ny)) - 1.0;
                d * min(half_x, half_y)
            } else {
                let r_right = if cpy > 0.0 { corner_radii.z } else { corner_radii.y };
                let r_left = if cpy > 0.0 { corner_radii.w } else { corner_radii.x };
                let r0 = if cpx > 0.0 { r_right } else { r_left };
                let r = min(r0, min(half_x, half_y));
                let qx = abs(cpx) - half_x + r;
                let qy = abs(cpy) - half_y + r;
                let outside = length(Vec2::new(max(qx, 0.0), max(qy, 0.0)));
                min(max(qx, qy), 0.0) + outside - r
            };
            let shape_alpha = 1.0 - smoothstep(-0.75, 0.75, dist);
            let fill = if border_width > 0.0 {
                let inner_alpha = 1.0 - smoothstep(-0.75, 0.75, dist + border_width);
                Vec4::new(mix(border_color.x, color.x, inner_alpha), mix(border_color.y, color.y, inner_alpha), mix(border_color.z, color.z, inner_alpha), mix(border_color.w, color.w, inner_alpha))
            } else {
                color
            };
            Vec4::new(fill.x * shape_alpha, fill.y * shape_alpha, fill.z * shape_alpha, fill.w * shape_alpha)
        }"#
        .into(),
    }
}

// ─── glyph ────────────────────────────────────────────────────────────────

fn glyph_vertex_def() -> ShaderDef {
    ShaderDef {
        name: "glyph_vertex_dsl".into(),
        stage: ShaderStage::Vertex,
        params: vec![
            p("pos", Vec2),
            p("corner", Vec2),
            p("size", Vec2),
            p("uv_rect", Vec4),
            p("color", Vec4),
            uniform("viewport", Vec2),
        ],
        return_type: Vec4,
        body_source: r#"{
            let px = pos.x + corner.x * size.x;
            let py = pos.y + corner.y * size.y;
            let ndc_x = px / viewport.x * 2.0 - 1.0;
            let ndc_y = 1.0 - py / viewport.y * 2.0;
            Vec4::new(ndc_x, ndc_y, 0.0, 1.0)
        }"#
        .into(),
    }
}

fn glyph_fragment_def() -> ShaderDef {
    ShaderDef {
        name: "glyph_fragment_dsl".into(),
        stage: ShaderStage::Fragment,
        // The `atlas: &Texture2D` param is not a ShaderParam — it lowers to a
        // texture slot; the body references it via `sample(0, uv)`.
        params: vec![p("corner", Vec2), p("uv_rect", Vec4), p("color", Vec4)],
        return_type: Vec4,
        body_source: r#"{
            let u = mix(uv_rect.x, uv_rect.z, corner.x);
            let v = mix(uv_rect.y, uv_rect.w, corner.y);
            let coverage = sample(0, Vec2::new(u, v)).x;
            Vec4::new(color.x * coverage, color.y * coverage, color.z * coverage, color.w * coverage)
        }"#
        .into(),
    }
}

// ─── image ────────────────────────────────────────────────────────────────

fn image_vertex_def() -> ShaderDef {
    ShaderDef {
        name: "image_vertex_dsl".into(),
        stage: ShaderStage::Vertex,
        params: vec![
            p("pos", Vec2),
            p("corner", Vec2),
            p("size", Vec2),
            p("uv_rect", Vec4),
            p("opacity", F32),
            p("border_radius", F32),
            uniform("viewport", Vec2),
        ],
        return_type: Vec4,
        body_source: r#"{
            let px = pos.x + corner.x * size.x;
            let py = pos.y + corner.y * size.y;
            let ndc_x = px / viewport.x * 2.0 - 1.0;
            let ndc_y = 1.0 - py / viewport.y * 2.0;
            Vec4::new(ndc_x, ndc_y, 0.0, 1.0)
        }"#
        .into(),
    }
}

fn image_fragment_def() -> ShaderDef {
    ShaderDef {
        name: "image_fragment_dsl".into(),
        stage: ShaderStage::Fragment,
        params: vec![
            p("corner", Vec2),
            p("size", Vec2),
            p("uv_rect", Vec4),
            p("opacity", F32),
            p("border_radius", F32),
        ],
        return_type: Vec4,
        body_source: r#"{
            let u = mix(uv_rect.x, uv_rect.z, corner.x);
            let v = mix(uv_rect.y, uv_rect.w, corner.y);
            let sampled = sample(0, Vec2::new(u, v));
            let color = Vec4::new(sampled.x * opacity, sampled.y * opacity, sampled.z * opacity, sampled.w * opacity);
            if border_radius > 0.0 {
                let px = corner.x * size.x;
                let py = corner.y * size.y;
                let half_x = size.x * 0.5;
                let half_y = size.y * 0.5;
                let r = min(border_radius, min(half_x, half_y));
                let cpx = px - half_x;
                let cpy = py - half_y;
                let qx = abs(cpx) - half_x + r;
                let qy = abs(cpy) - half_y + r;
                let outside = length(Vec2::new(max(qx, 0.0), max(qy, 0.0)));
                let dist = min(max(qx, qy), 0.0) + outside - r;
                let mask = 1.0 - smoothstep(-0.75, 0.75, dist);
                Vec4::new(color.x * mask, color.y * mask, color.z * mask, color.w * mask)
            } else {
                color
            }
        }"#
        .into(),
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[test]
fn rect_vertex_metallib() {
    if !xcrun_present() {
        eprintln!("SKIP rect_vertex_metallib: no xcrun");
        return;
    }
    let msl = emit_vertex_shader(&rect_vertex_def()).unwrap();
    // Uniform binds at [[buffer(0)]] (UNIFORM_SLOT_DSL) — the ABI the runtime expects.
    assert!(
        msl.contains("constant float2& viewport [[buffer(0)]]"),
        "ABI: {msl}"
    );
    must_compile(&msl, "rect_vertex");
}

#[test]
fn rect_fragment_metallib() {
    if !xcrun_present() {
        eprintln!("SKIP rect_fragment_metallib: no xcrun");
        return;
    }
    let msl = emit_fragment_shader(&rect_fragment_def()).unwrap();
    must_compile(&msl, "rect_fragment");
}

#[test]
fn glyph_vertex_metallib() {
    if !xcrun_present() {
        eprintln!("SKIP glyph_vertex_metallib: no xcrun");
        return;
    }
    let msl = emit_vertex_shader(&glyph_vertex_def()).unwrap();
    must_compile(&msl, "glyph_vertex");
}

#[test]
fn glyph_fragment_metallib() {
    if !xcrun_present() {
        eprintln!("SKIP glyph_fragment_metallib: no xcrun");
        return;
    }
    let msl = emit_fragment_shader(&glyph_fragment_def()).unwrap();
    // sample(0, uv) must have been rewritten to the texture form.
    assert!(
        msl.contains("tex_0.sample(smp_0,"),
        "texture rewrite: {msl}"
    );
    assert!(
        msl.contains("texture2d<float> tex_0 [[texture(0)]]"),
        "tex binding: {msl}"
    );
    must_compile(&msl, "glyph_fragment");
}

#[test]
fn image_vertex_metallib() {
    if !xcrun_present() {
        eprintln!("SKIP image_vertex_metallib: no xcrun");
        return;
    }
    let msl = emit_vertex_shader(&image_vertex_def()).unwrap();
    must_compile(&msl, "image_vertex");
}

#[test]
fn image_fragment_metallib() {
    if !xcrun_present() {
        eprintln!("SKIP image_fragment_metallib: no xcrun");
        return;
    }
    let msl = emit_fragment_shader(&image_fragment_def()).unwrap();
    // Trailing statement-position `if/else` as the fragment's return value.
    assert!(msl.contains("if (border_radius > 0.0) {"), "tail-if: {msl}");
    assert!(
        msl.contains("tex_0.sample(smp_0,"),
        "texture rewrite: {msl}"
    );
    must_compile(&msl, "image_fragment");
}
