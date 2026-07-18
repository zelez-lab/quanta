//! Integration tests: full vertex/fragment MSL emission + metallib compile.
//!
//! The unit tests in `shader_ast_tests.rs` assert on emitted MSL text; these
//! close the loop by feeding the emitted MSL to `xcrun metal`/`metallib` (the
//! same path the shader pipeline uses) and asserting the driver accepts it.
//! They gate on `xcrun` presence like the existing metallib tests, so they
//! `SKIP` on non-Apple hosts / hosts without the Metal toolchain.
//!
//! The three fixtures are the dija render shaders (rect / glyph / image),
//! rewritten to the shared-struct varying model: each pair declares ONE
//! Varyings struct, the vertex returns it (tail struct literal), and the
//! fragment consumes it through a receiver param (`s.<field>`). Each had
//! NEVER produced a working metallib before this emitter (the consumer fell
//! back to hand-written MSL); these lock in that they still do under the
//! explicit interface.
//!
//! Fixture bodies are written in clean Rust source. `syn` parses that
//! identically to the token-stringified form the proc macro actually ships
//! (verified: `Vec4 :: new` and `Vec4::new` parse to the same AST), so the
//! fixtures stay readable while exercising the real emit path. Texture fragment
//! bodies use the canonical `sample(N, uv)` slot form the macro's
//! `rewrite_texture_names` produces before the wire — that's what the compiler
//! receives.

use quanta_ir::{ShaderDef, ShaderParam, ShaderStage, ShaderType, ShaderVaryings, VaryingField};

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
        is_slice: false,
    }
}

fn uniform(name: &str, ty: ShaderType) -> ShaderParam {
    ShaderParam {
        name: name.to_string(),
        ty,
        is_uniform: true,
        is_slice: false,
    }
}

fn field(name: &str, ty: ShaderType) -> VaryingField {
    VaryingField {
        name: name.to_string(),
        ty,
    }
}

/// The shared interface of a pair: position field `clip` + the given
/// varyings; `binding` distinguishes the vertex (None) from the fragment
/// (`Some(receiver)`) side.
fn varyings(
    struct_name: &str,
    fields: &[(&str, ShaderType)],
    binding: Option<&str>,
) -> ShaderVaryings {
    ShaderVaryings {
        struct_name: struct_name.to_string(),
        position: "clip".to_string(),
        fields: fields.iter().map(|(n, t)| field(n, *t)).collect(),
        binding: binding.map(str::to_string),
    }
}

/// Emit MSL, then compile it to metallib; assert the driver accepts it.
fn must_compile(msl: &str, label: &str) {
    match compile_msl_to_metallib(msl) {
        Ok(Some(bytes)) => assert!(!bytes.is_empty(), "{label}: empty metallib"),
        Ok(None) => panic!("{label}: xcrun unexpectedly absent after presence check"),
        Err(e) => panic!("{label}: metallib compile failed:\n{e}\n--- MSL ---\n{msl}"),
    }
}

use ShaderType::{F32, Vec2, Vec4};

// ─── rect ─────────────────────────────────────────────────────────────────

const RECT_FIELDS: &[(&str, ShaderType)] = &[
    ("corner", Vec2),
    ("size", Vec2),
    ("color", Vec4),
    ("border_color", Vec4),
    ("corner_radii", Vec4),
    ("border_width", F32),
    ("shape_type", F32),
];

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
            RectSurface {
                clip: Vec4::new(ndc_x, ndc_y, 0.0, 1.0),
                corner: corner,
                size: size,
                color: color,
                border_color: border_color,
                corner_radii: corner_radii,
                border_width: border_width,
                shape_type: shape_type,
            }
        }"#
        .into(),
        varyings: Some(varyings("RectSurface", RECT_FIELDS, None)),
    }
}

fn rect_fragment_def() -> ShaderDef {
    ShaderDef {
        name: "rect_fragment".into(),
        stage: ShaderStage::Fragment,
        params: vec![],
        return_type: Vec4,
        body_source: r#"{
            let px = s.corner.x * s.size.x;
            let py = s.corner.y * s.size.y;
            let half_x = s.size.x * 0.5;
            let half_y = s.size.y * 0.5;
            let cpx = px - half_x;
            let cpy = py - half_y;
            let dist = if s.shape_type > 0.5 {
                let nx = cpx / half_x;
                let ny = cpy / half_y;
                let d = length(Vec2::new(nx, ny)) - 1.0;
                d * min(half_x, half_y)
            } else {
                let r_right = if cpy > 0.0 { s.corner_radii.z } else { s.corner_radii.y };
                let r_left = if cpy > 0.0 { s.corner_radii.w } else { s.corner_radii.x };
                let r0 = if cpx > 0.0 { r_right } else { r_left };
                let r = min(r0, min(half_x, half_y));
                let qx = abs(cpx) - half_x + r;
                let qy = abs(cpy) - half_y + r;
                let outside = length(Vec2::new(max(qx, 0.0), max(qy, 0.0)));
                min(max(qx, qy), 0.0) + outside - r
            };
            let shape_alpha = 1.0 - smoothstep(-0.75, 0.75, dist);
            let fill = if s.border_width > 0.0 {
                let inner_alpha = 1.0 - smoothstep(-0.75, 0.75, dist + s.border_width);
                Vec4::new(mix(s.border_color.x, s.color.x, inner_alpha), mix(s.border_color.y, s.color.y, inner_alpha), mix(s.border_color.z, s.color.z, inner_alpha), mix(s.border_color.w, s.color.w, inner_alpha))
            } else {
                s.color
            };
            Vec4::new(fill.x * shape_alpha, fill.y * shape_alpha, fill.z * shape_alpha, fill.w * shape_alpha)
        }"#
        .into(),
        varyings: Some(varyings("RectSurface", RECT_FIELDS, Some("s"))),
    }
}

// ─── glyph ────────────────────────────────────────────────────────────────

const GLYPH_FIELDS: &[(&str, ShaderType)] = &[("corner", Vec2), ("uv_rect", Vec4), ("color", Vec4)];

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
            GlyphSurface {
                clip: Vec4::new(ndc_x, ndc_y, 0.0, 1.0),
                corner: corner,
                uv_rect: uv_rect,
                color: color,
            }
        }"#
        .into(),
        varyings: Some(varyings("GlyphSurface", GLYPH_FIELDS, None)),
    }
}

fn glyph_fragment_def() -> ShaderDef {
    ShaderDef {
        name: "glyph_fragment_dsl".into(),
        stage: ShaderStage::Fragment,
        // The `atlas: &Texture2D` param is not a ShaderParam — it lowers to a
        // texture slot; the body references it via `sample(0, uv)`.
        params: vec![],
        return_type: Vec4,
        body_source: r#"{
            let u = mix(s.uv_rect.x, s.uv_rect.z, s.corner.x);
            let v = mix(s.uv_rect.y, s.uv_rect.w, s.corner.y);
            let coverage = sample(0, Vec2::new(u, v)).x;
            Vec4::new(s.color.x * coverage, s.color.y * coverage, s.color.z * coverage, s.color.w * coverage)
        }"#
        .into(),
        varyings: Some(varyings("GlyphSurface", GLYPH_FIELDS, Some("s"))),
    }
}

// ─── image ────────────────────────────────────────────────────────────────

const IMAGE_FIELDS: &[(&str, ShaderType)] = &[
    ("corner", Vec2),
    ("size", Vec2),
    ("uv_rect", Vec4),
    ("opacity", F32),
    ("border_radius", F32),
];

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
            ImageSurface {
                clip: Vec4::new(ndc_x, ndc_y, 0.0, 1.0),
                corner: corner,
                size: size,
                uv_rect: uv_rect,
                opacity: opacity,
                border_radius: border_radius,
            }
        }"#
        .into(),
        varyings: Some(varyings("ImageSurface", IMAGE_FIELDS, None)),
    }
}

fn image_fragment_def() -> ShaderDef {
    ShaderDef {
        name: "image_fragment_dsl".into(),
        stage: ShaderStage::Fragment,
        params: vec![],
        return_type: Vec4,
        body_source: r#"{
            let u = mix(s.uv_rect.x, s.uv_rect.z, s.corner.x);
            let v = mix(s.uv_rect.y, s.uv_rect.w, s.corner.y);
            let sampled = sample(0, Vec2::new(u, v));
            let color = Vec4::new(sampled.x * s.opacity, sampled.y * s.opacity, sampled.z * s.opacity, sampled.w * s.opacity);
            if s.border_radius > 0.0 {
                let px = s.corner.x * s.size.x;
                let py = s.corner.y * s.size.y;
                let half_x = s.size.x * 0.5;
                let half_y = s.size.y * 0.5;
                let r = min(s.border_radius, min(half_x, half_y));
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
        varyings: Some(varyings("ImageSurface", IMAGE_FIELDS, Some("s"))),
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
    // The stage-out struct is the shared Varyings struct: [[position]] on the
    // position field, [[user(locN)]] per varying, every member assigned
    // explicitly from the tail literal.
    assert!(
        msl.contains("float4 clip [[position]];"),
        "position member: {msl}"
    );
    assert!(msl.contains("float2 corner [[user(loc0)]];"), "loc0: {msl}");
    assert!(
        msl.contains("out.corner = corner;"),
        "explicit store: {msl}"
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
    // The stage-in struct mirrors the vertex out struct member for member.
    assert!(
        msl.contains("RectSurface s [[stage_in]]"),
        "receiver: {msl}"
    );
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
    assert!(
        msl.contains("if (s.border_radius > 0.0) {"),
        "tail-if: {msl}"
    );
    assert!(
        msl.contains("tex_0.sample(smp_0,"),
        "texture rewrite: {msl}"
    );
    must_compile(&msl, "image_fragment");
}
