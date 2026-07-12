#![cfg(feature = "render")]
//! Live-GPU half of the shader parity corpus.
//!
//! Every draw here mirrors a fixture in `quanta-compiler`'s
//! `shader_parity_tests` (cross-referenced by fixture name in a comment above
//! each shader). Layer A proves the two native emitters AGREE on what a body
//! means structurally; these draws prove the emitted shader actually produces
//! the right pixels on a real GPU. Pixels are asserted with a per-channel
//! tolerance of 2.
//!
//! Orientation contract: Metal's NDC is y-up, Vulkan's is y-down, so the same
//! quad renders vertically flipped between backends. Every assertion here is
//! therefore either (a) vertically symmetric, (b) varying along x only, or
//! (c) uses the corner-probe branch that detects the flip from one pixel and
//! then asserts the full layout. No test assumes a y direction.

use quanta::RenderGpu;

use quanta::render_pass::ColorTarget;
use quanta::{Color, FieldUsage, Format, LoadOp, StoreOp};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn pixel_at(pixels: &[u8], w: u32, x: u32, y: u32) -> (u8, u8, u8, u8) {
    let i = ((y * w + x) * 4) as usize;
    (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3])
}

/// Assert a pixel matches `want` within a per-channel tolerance of 2.
fn expect_rgb(pixels: &[u8], w: u32, x: u32, y: u32, want: (u8, u8, u8), which: &str) {
    let (r, g, b, _) = pixel_at(pixels, w, x, y);
    assert!(
        r.abs_diff(want.0) <= 2 && g.abs_diff(want.1) <= 2 && b.abs_diff(want.2) <= 2,
        "{which} at ({x},{y}): expected {want:?}, got ({r},{g},{b})"
    );
}

// ─── Shared geometry ─────────────────────────────────────────────────────────

fn pos_uv_layout() -> Vec<quanta::VertexLayout> {
    vec![quanta::VertexLayout {
        stride: 20, // 3 floats pos + 2 floats uv
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

/// Fullscreen quad, pos(x,y,z) + uv(u,v), two triangles covering [-1,1].
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

fn field_of(gpu: &quanta::Gpu, data: &[f32]) -> quanta::Field<f32> {
    let f: quanta::Field<f32> = gpu
        .field_with_usage(data.len(), FieldUsage::default_render())
        .unwrap();
    f.write(data).unwrap();
    f
}

// ─── Shaders (bodies mirror the named Layer-A fixtures) ──────────────────────

// Standard fullscreen-quad vertex (mirrors the vertex given to D1): pass the
// clip-space position straight through so uv interpolates across the target.
#[quanta::vertex]
fn quad_vertex(pos: Vec3, uv: Vec2) -> Vec4 {
    Vec4::new(pos.x, pos.y, 0.0, 1.0)
}

// mirrors fixture `expr_if_nested`
#[quanta::fragment]
fn nested_expr_if_frag(uv: Vec2) -> Vec4 {
    let c = if uv.x < 0.25 {
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    } else {
        if uv.x < 0.5 {
            Vec4::new(0.0, 1.0, 0.0, 1.0)
        } else {
            if uv.x < 0.75 {
                Vec4::new(0.0, 0.0, 1.0, 1.0)
            } else {
                Vec4::new(1.0, 1.0, 1.0, 1.0)
            }
        }
    };
    c
}

// mirrors fixture `stmt_if_nested`
#[quanta::fragment]
fn nested_stmt_if_frag(uv: Vec2) -> Vec4 {
    let mut c = Vec4::new(1.0, 1.0, 1.0, 1.0);
    if uv.x < 0.25 {
        c = Vec4::new(1.0, 0.0, 0.0, 1.0);
    } else {
        if uv.x < 0.5 {
            c = Vec4::new(0.0, 1.0, 0.0, 1.0);
        } else {
            if uv.x < 0.75 {
                c = Vec4::new(0.0, 0.0, 1.0, 1.0);
            } else {
            }
        }
    }
    c
}

// mirrors fixture `swizzle_multi`: recombine channels from a computed Vec4.
#[quanta::fragment]
fn multi_swizzle_frag(uv: Vec2) -> Vec4 {
    let base = Vec4::new(0.2, 0.4, 0.6, 0.8);
    let s = base.zw;
    Vec4::new(s.x, s.y, base.xy.x, 1.0)
}

// mirrors fixture `sample_swizzle`: sample .x times a uniform tint.
#[quanta::fragment]
fn sample_swizzle_frag(uv: Vec2, tint: &Vec4) -> Vec4 {
    let c = sample(0, uv).x;
    Vec4::new(c * tint.x, c * tint.y, c * tint.z, 1.0)
}

// mirrors fixture `smoothstep_band`: fwidth-scaled circle SDF band. The
// coverage is written into RGB (not just alpha) so a no-blend readback reads
// it directly from the R channel.
#[quanta::fragment]
fn fwidth_circle_frag(uv: Vec2) -> Vec4 {
    let dx = uv.x - 0.5;
    let dy = uv.y - 0.5;
    let d = length(Vec2::new(dx, dy)) - 0.3;
    let w = fwidth(d);
    let a = 1.0 - smoothstep(-w, w, d);
    Vec4::new(a, a, a, 1.0)
}

// mirrors fixture `uniform_two_frag`: non-commutative combiner so binding
// order is observable. mix(base, gain, 0.25) = 0.75*base + 0.25*gain.
// Kept as short single-line statements: a rustfmt-wrapped multi-line call
// ships a trailing comma in the token stream, which the SPIR-V argument
// parser rejects (pinned by the `ctor_trailing_comma` fixture) — the shader
// would silently fall to a passthrough on Vulkan.
#[quanta::fragment]
fn two_uniform_frag(uv: Vec2, base: &Vec4, gain: &Vec4) -> Vec4 {
    let r = mix(base.x, gain.x, 0.25);
    let g = mix(base.y, gain.y, 0.25);
    let b = mix(base.z, gain.z, 0.25);
    Vec4::new(r, g, b, 1.0)
}

// mirrors fixture `uniform_deref_star`: both deref forms in one body.
#[quanta::vertex]
fn deref_forms_vertex(pos: Vec3, uv: Vec2, shift: &Vec2) -> Vec4 {
    Vec4::new(pos.x + (*shift).x, pos.y + shift.y, 0.0, 1.0)
}

#[quanta::fragment]
fn solid_red_frag() -> Vec4 {
    Vec4::new(1.0, 0.0, 0.0, 1.0)
}

// Shared-slot vertex/fragment: both read uniform 0 (a &Vec2 `off`).
#[quanta::vertex]
fn shared_off_vertex(pos: Vec3, uv: Vec2, off: &Vec2) -> Vec4 {
    Vec4::new(pos.x + (*off).x, pos.y, 0.0, 1.0)
}

#[quanta::fragment]
fn shared_off_frag(uv: Vec2, off: &Vec2) -> Vec4 {
    Vec4::new(off.x, off.y, 0.0, 1.0)
}

// mirrors fixture `arith_mix`: a horizontal ramp, live varying interpolation.
#[quanta::fragment]
fn horizontal_ramp_frag(uv: Vec2) -> Vec4 {
    Vec4::new(uv.x, 1.0 - uv.x, 0.0, 1.0)
}

// mirrors fixture `dija_rect_frag`: the real rounded-rect signed-distance body
// from rect_fragment_def in emit_msl/shader_tests.rs. `corner` arrives as the
// interpolated uv varying (0..1 across the quad); the shape parameters are
// uniforms so a single quad_vertex can drive it. `size`/`border_width` carry
// their scalars in `.x` (the DSL exposes uniforms as vectors).
#[quanta::fragment]
fn dija_rect_frag(
    corner: Vec2,
    size: &Vec2,
    color: &Vec4,
    border_color: &Vec4,
    corner_radii: &Vec4,
    border_width: &Vec2,
) -> Vec4 {
    let sx = (*size).x;
    let sy = (*size).y;
    let px = corner.x * sx;
    let py = corner.y * sy;
    let half_x = sx * 0.5;
    let half_y = sy * 0.5;
    let cpx = px - half_x;
    let cpy = py - half_y;
    let r_right = if cpy > 0.0 {
        (*corner_radii).z
    } else {
        (*corner_radii).y
    };
    let r_left = if cpy > 0.0 {
        (*corner_radii).w
    } else {
        (*corner_radii).x
    };
    let r0 = if cpx > 0.0 { r_right } else { r_left };
    let r = min(r0, min(half_x, half_y));
    let qx = abs(cpx) - half_x + r;
    let qy = abs(cpy) - half_y + r;
    let outside = length(Vec2::new(max(qx, 0.0), max(qy, 0.0)));
    let dist = min(max(qx, qy), 0.0) + outside - r;
    let shape_alpha = 1.0 - smoothstep(-0.75, 0.75, dist);
    let inner_alpha = 1.0 - smoothstep(-0.75, 0.75, dist + (*border_width).x);
    let fill_x = mix((*border_color).x, (*color).x, inner_alpha);
    let fill_y = mix((*border_color).y, (*color).y, inner_alpha);
    let fill_z = mix((*border_color).z, (*color).z, inner_alpha);
    // Short single-line tail: a rustfmt-wrapped call ships a trailing comma,
    // which falls to a SPIR-V passthrough (see `ctor_trailing_comma`).
    let out_x = fill_x * shape_alpha;
    let out_y = fill_y * shape_alpha;
    let out_z = fill_z * shape_alpha;
    Vec4::new(out_x, out_y, out_z, 1.0)
}

// mirrors fixture `uniform_mat4_mul`: translate-only matrix moves the quad.
#[quanta::vertex]
fn mvp_vertex(pos: Vec3, uv: Vec2, mvp: &Mat4) -> Vec4 {
    mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

// ─── Pipeline helper ─────────────────────────────────────────────────────────

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
        .all(|b| b.for_vendor(gpu.caps().vendor).is_some())
}

// ─── D1: nested expression-if, 4 colour bands on uv.x ────────────────────────

#[test]
fn draw_nested_expression_if() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&QUAD_VERTEX_SHADER, &NESTED_EXPR_IF_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &QUAD_VERTEX_SHADER, &NESTED_EXPR_IF_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    let w = 8u32;
    let h = 8u32;
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

    // Sample one column per band, at BOTH y extremes — the band boundaries are
    // on uv.x only, so the expected colour is flip-independent.
    for &y in &[0u32, h - 1] {
        expect_rgb(&px, w, 1, y, (255, 0, 0), "band0 red");
        expect_rgb(&px, w, 3, y, (0, 255, 0), "band1 green");
        expect_rgb(&px, w, 5, y, (0, 0, 255), "band2 blue");
        expect_rgb(&px, w, 7, y, (255, 255, 255), "band3 white");
    }
}

// ─── D2: nested statement-if (OpPhi chain), same 4 bands ─────────────────────

#[test]
fn draw_nested_statement_if() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&QUAD_VERTEX_SHADER, &NESTED_STMT_IF_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &QUAD_VERTEX_SHADER, &NESTED_STMT_IF_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    let w = 8u32;
    let h = 8u32;
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

    for &y in &[0u32, h - 1] {
        expect_rgb(&px, w, 1, y, (255, 0, 0), "band0 red");
        expect_rgb(&px, w, 3, y, (0, 255, 0), "band1 green");
        expect_rgb(&px, w, 5, y, (0, 0, 255), "band2 blue");
        expect_rgb(&px, w, 7, y, (255, 255, 255), "band3 white");
    }
}

// ─── D3: multi-swizzle recombination, constant colour everywhere ─────────────

#[test]
fn draw_multi_swizzle() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&QUAD_VERTEX_SHADER, &MULTI_SWIZZLE_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &QUAD_VERTEX_SHADER, &MULTI_SWIZZLE_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    let w = 8u32;
    let h = 8u32;
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

    // base = (0.2,0.4,0.6,0.8); s = base.zw = (0.6,0.8); out = (s.x, s.y,
    // base.xy.x, 1) = (0.6, 0.8, 0.2, 1) → (153, 204, 51). Constant, so probe
    // centre and two corners.
    let want = (153, 204, 51);
    expect_rgb(&px, w, 4, 4, want, "centre");
    expect_rgb(&px, w, 0, 0, want, "corner00");
    expect_rgb(&px, w, 7, 7, want, "corner77");
}

// ─── D4: sample .x modulated by a uniform tint, per-quadrant colours ─────────

#[test]
fn draw_sample_swizzle_modulate() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&QUAD_VERTEX_SHADER, &SAMPLE_SWIZZLE_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &QUAD_VERTEX_SHADER, &SAMPLE_SWIZZLE_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    // 2×2 luminance-in-R texture: the sampled .x per quadrant is 1.0, 0.5,
    // 0.25, 0.0. tint = (1, 0.5, 0.25, 1) → each quadrant is c*(tint.xyz).
    let tex_data: [u8; 16] = [
        255, 0, 0, 255, // (0,0) x=1.0
        128, 0, 0, 255, // (1,0) x=0.5
        64, 0, 0, 255, // (0,1) x=0.25
        0, 0, 0, 255, // (1,1) x=0.0
    ];
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(2, 2, Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ),
        )
        .expect("texture");
    tex.write(&tex_data).expect("tex write");
    let tint = field_of(&gpu, &[1.0, 0.5, 0.25, 1.0]);

    let w = 4u32;
    let h = 4u32;
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
        .uniform(0, &tint)
        .texture(0, &tex)
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

    // Per quadrant: c * (1, 0.5, 0.25). c=1.0→(255,128,64); 0.5→(128,64,32);
    // 0.25→(64,32,16); 0.0→(0,0,0). Detect vertical orientation from one
    // corner (uv(0,0) has c=1.0, the brightest), then assert the full layout.
    let (r_bl, _, _, _) = pixel_at(&px, w, 0, 3);
    if r_bl > 128 {
        // Metal-style: v=0 (uv(0,0), c=1.0) at the bottom rows.
        expect_rgb(&px, w, 0, 3, (255, 128, 64), "uv(0,0) c=1.0");
        expect_rgb(&px, w, 3, 3, (128, 64, 32), "uv(1,0) c=0.5");
        expect_rgb(&px, w, 0, 0, (64, 32, 16), "uv(0,1) c=0.25");
        expect_rgb(&px, w, 3, 0, (0, 0, 0), "uv(1,1) c=0.0");
    } else {
        // Vulkan-style: vertically flipped.
        expect_rgb(&px, w, 0, 0, (255, 128, 64), "uv(0,0) c=1.0");
        expect_rgb(&px, w, 3, 0, (128, 64, 32), "uv(1,0) c=0.5");
        expect_rgb(&px, w, 0, 3, (64, 32, 16), "uv(0,1) c=0.25");
        expect_rgb(&px, w, 3, 3, (0, 0, 0), "uv(1,1) c=0.0");
    }
}

// ─── D5: fwidth-scaled circle band — inside, outside, and a ring pixel ───────

#[test]
fn draw_fwidth_circle_band() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&QUAD_VERTEX_SHADER, &FWIDTH_CIRCLE_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &QUAD_VERTEX_SHADER, &FWIDTH_CIRCLE_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    // 16×16 so the smoothstep band (width ~= fwidth(d), a few texels) lands on
    // real intermediate pixels rather than snapping fully on/off.
    let w = 16u32;
    let h = 16u32;
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

    // The shader outputs white pre-multiplied by alpha over a black clear, so
    // the visible R (== alpha*255) reads the coverage directly. Circle is
    // radius 0.3 in uv centred at (0.5,0.5) → symmetric, orientation-free.
    let (rc, _, _, _) = pixel_at(&px, w, 8, 8); // centre: fully inside
    assert!(rc > 230, "centre must be inside the disc (R={rc})");
    let (ro, _, _, _) = pixel_at(&px, w, 0, 0); // corner: fully outside
    assert!(ro < 25, "corner must be outside the disc (R={ro})");

    // At least one pixel on the ring must be strictly intermediate.
    let mut ring = None;
    for y in 0..h {
        for x in 0..w {
            let (r, _, _, _) = pixel_at(&px, w, x, y);
            if (13..=242).contains(&r) {
                ring = Some((x, y, r));
            }
        }
    }
    assert!(
        ring.is_some(),
        "expected at least one antialiased ring pixel (0.05 < a < 0.95)"
    );
}

// ─── D6: two uniforms, order-sensitive combiner ──────────────────────────────

#[test]
fn draw_two_uniforms_binding_order() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&QUAD_VERTEX_SHADER, &TWO_UNIFORM_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &QUAD_VERTEX_SHADER, &TWO_UNIFORM_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    // mix(base, gain, 0.25) = 0.75*base + 0.25*gain. base=(1,0,0,1),
    // gain=(0,1,0,1) → (0.75, 0.25, 0, 1) → (191, 64, 0). Swapping the binds
    // would give mix(gain, base, 0.25) = (0.25, 0.75, 0) → (64, 191, 0), a
    // clearly different colour.
    let base = field_of(&gpu, &[1.0, 0.0, 0.0, 1.0]);
    let gain = field_of(&gpu, &[0.0, 1.0, 0.0, 1.0]);

    let w = 8u32;
    let h = 8u32;
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
        .uniform(0, &base)
        .uniform(1, &gain)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
    let px = target.read().unwrap();

    // Constant colour; the value pins that base bound at 0 and gain at 1.
    expect_rgb(&px, w, 4, 4, (191, 64, 0), "mix(base,gain,0.25)");
}

// ─── D7: both deref forms in one vertex body, x-shift ────────────────────────

#[test]
fn draw_deref_forms() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&DEREF_FORMS_VERTEX_SHADER, &SOLID_RED_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &DEREF_FORMS_VERTEX_SHADER, &SOLID_RED_FRAG_SHADER);

    // Fullscreen quad shifted +1.0 in x, so only the right half of the screen
    // stays covered. Exercises BOTH deref forms: x via (*shift).x and y via
    // shift.y (0 here). x-only → orientation-free.
    let vb = fullscreen_vb(&gpu);
    let shift = field_of(&gpu, &[1.0, 0.0]);

    let w = 8u32;
    let h = 8u32;
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
        .uniform(0, &shift)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
    let px = target.read().unwrap();

    // The fullscreen quad is shifted +1.0 → visible only for screen x in the
    // right half; the left half is uncovered (clear).
    for &y in &[0u32, h - 1] {
        let (rl, _, _, _) = pixel_at(&px, w, 1, y);
        assert!(rl < 30, "left half must be clear at y={y} (R={rl})");
        expect_rgb(&px, w, 6, y, (255, 0, 0), "right half red");
    }
}

// ─── D8: one uniform slot feeds both stages ──────────────────────────────────

#[test]
fn draw_shared_uniform_slot_across_stages() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&SHARED_OFF_VERTEX_SHADER, &SHARED_OFF_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &SHARED_OFF_VERTEX_SHADER, &SHARED_OFF_FRAG_SHADER);

    // Fullscreen quad; off=(1.0, 0.6). The vertex shifts x by +1.0 (→ only the
    // right half stays covered), the fragment colours (off.x, off.y, 0) =
    // (1.0, 0.6, 0). ONE .uniform(0) bind feeds both stages.
    let vb = fullscreen_vb(&gpu);
    let off = field_of(&gpu, &[1.0, 0.6]);

    let w = 8u32;
    let h = 8u32;
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
        .uniform(0, &off)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
    let px = target.read().unwrap();

    // Right half shows (1.0, 0.6, 0) → (255, 153, 0); left half clear. Both the
    // shift (vertex read) and the colour (fragment read) must hold — proving
    // one bound uniform reaches both stages.
    for &y in &[0u32, h - 1] {
        let (rl, gl, _, _) = pixel_at(&px, w, 1, y);
        assert!(rl < 30 && gl < 30, "left half must be clear at y={y}");
        expect_rgb(&px, w, 6, y, (255, 153, 0), "right half (off.x, off.y, 0)");
    }
}

// ─── D9: horizontal ramp — arithmetic + varying interpolation ────────────────

#[test]
fn draw_horizontal_ramp() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&QUAD_VERTEX_SHADER, &HORIZONTAL_RAMP_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &QUAD_VERTEX_SHADER, &HORIZONTAL_RAMP_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    let w = 8u32;
    let h = 8u32;
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

    // out = (uv.x, 1-uv.x, 0). uv.x rises left→right, so R climbs and G falls.
    // Varies on x only → orientation-free. Assert the monotonic trend at both
    // y extremes rather than exact texel-centre values.
    for &y in &[0u32, h - 1] {
        let (rl, gl, _, _) = pixel_at(&px, w, 0, y);
        let (rr, gr, _, _) = pixel_at(&px, w, 7, y);
        assert!(rl < 60, "left is green-ish, low R (R={rl})");
        assert!(gl > 195, "left is green-ish, high G (G={gl})");
        assert!(rr > 195, "right is red-ish, high R (R={rr})");
        assert!(gr < 60, "right is red-ish, low G (G={gr})");
    }
}

// ─── D10: the real dija rounded rect with border ─────────────────────────────

#[test]
fn draw_dija_rect_body() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&QUAD_VERTEX_SHADER, &DIJA_RECT_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &QUAD_VERTEX_SHADER, &DIJA_RECT_FRAG_SHADER);
    let vb = fullscreen_vb(&gpu);

    // corner arrives as uv (0..1 across the quad). A 32×32 shape with a green
    // fill, a distinct red border a few px wide, and generous corner radii.
    let size = field_of(&gpu, &[32.0, 32.0]);
    let color = field_of(&gpu, &[0.2, 0.8, 0.2, 1.0]); // green fill
    let border_color = field_of(&gpu, &[0.9, 0.1, 0.1, 1.0]); // red border
    let corner_radii = field_of(&gpu, &[6.0, 6.0, 6.0, 6.0]);
    let border_width = field_of(&gpu, &[3.0, 0.0]);

    let w = 32u32;
    let h = 32u32;
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
        .uniform(0, &size)
        .uniform(1, &color)
        .uniform(2, &border_color)
        .uniform(3, &corner_radii)
        .uniform(4, &border_width)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
    let px = target.read().unwrap();

    // Points chosen on the horizontal midline / symmetric → orientation-free.
    // Centre = fill (green). A corner pixel lies outside the rounded corner
    // radius → clear. An edge-midpoint band is border (red-dominant).
    let (rc, gc, bc, _) = pixel_at(&px, w, w / 2, h / 2);
    assert!(
        gc > 150 && rc < 120 && bc < 120,
        "centre must be green fill, got ({rc},{gc},{bc})"
    );
    let (rk, gk, bk, _) = pixel_at(&px, w, 0, 0);
    assert!(
        rk < 40 && gk < 40 && bk < 40,
        "outer corner must be clear (outside radius), got ({rk},{gk},{bk})"
    );
    // Left edge midpoint (x=0 midline) is inside the shape but within the
    // border band → red-dominant.
    let (re, ge, _, _) = pixel_at(&px, w, 0, h / 2);
    assert!(
        re > ge,
        "left edge midpoint must be border (red > green), got R={re} G={ge}"
    );
}

// ─── D11: translate-only Mat4 uniform moves the quad to the right half ───────

#[test]
fn draw_mat4_translate() {
    let Some(gpu) = try_gpu() else { return };
    if !shaders_ready(&gpu, &[&MVP_VERTEX_SHADER, &SOLID_RED_FRAG_SHADER]) {
        eprintln!("SKIP: no shader binary");
        return;
    }
    let pipe = pipeline(&gpu, &MVP_VERTEX_SHADER, &SOLID_RED_FRAG_SHADER);

    // Small centred quad x[-0.4,0.4]; a column-major translate-only matrix
    // shifts +0.5 in x → the quad lands in the right half. x-only → flip-free.
    // (Distinct from the existing mat4 test, which SCALES.)
    #[rustfmt::skip]
    let verts: [f32; 30] = [
        -0.4, -0.4, 0.0,  0.0, 0.0,
         0.4, -0.4, 0.0,  1.0, 0.0,
         0.4,  0.4, 0.0,  1.0, 1.0,
        -0.4, -0.4, 0.0,  0.0, 0.0,
         0.4,  0.4, 0.0,  1.0, 1.0,
        -0.4,  0.4, 0.0,  0.0, 1.0,
    ];
    let vb = field_of(&gpu, &verts);
    #[rustfmt::skip]
    let mvp: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.5, 0.0, 0.0, 1.0, // translation column (column-major): +0.5 x
    ];
    let mvp_field = field_of(&gpu, &mvp);

    let w = 8u32;
    let h = 8u32;
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
        .uniform(0, &mvp_field)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();
    let px = target.read().unwrap();

    // The quad spans x in [0.1, 0.9] after the +0.5 shift → right half red,
    // left edge clear.
    for &y in &[3u32, 4] {
        let (rl, _, _, _) = pixel_at(&px, w, 0, y);
        assert!(
            rl < 30,
            "left edge must be clear after +0.5 x shift (R={rl})"
        );
        let (rr, _, _, _) = pixel_at(&px, w, 6, y);
        assert!(rr > 200, "right half must be covered (R={rr})");
    }
}
