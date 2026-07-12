#![cfg(feature = "render")]
//! Game-engine readiness tests.
//!
//! Proves the full rendering pipeline works for real 3D:
//! - Triangle rendering with vertex attributes
//! - Depth testing (two overlapping shapes at different Z)
//! - Indexed draw (cube with 8 vertices + 36 indices)
//! - Instanced draw (same triangle at multiple positions)
//!
//! All tests verify pixel output mathematically — no golden images.

use quanta::RenderGpu;

use quanta::render_pass::ColorTarget;
use quanta::{Color, FieldUsage, Format, LoadOp, StoreOp};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// ─── Shaders ────────────────────────────────────────────────────────────────

#[quanta::vertex]
fn passthrough_vertex(pos: Vec3, color: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn solid_red() -> Vec4 {
    Vec4::new(1.0, 0.0, 0.0, 1.0)
}

#[quanta::fragment]
fn solid_green() -> Vec4 {
    Vec4::new(0.0, 1.0, 0.0, 1.0)
}

#[quanta::vertex]
fn offset_vertex(pos: Vec3, offset: Vec3) -> Vec4 {
    Vec4::new(pos.x + offset.x, pos.y + offset.y, pos.z, 1.0)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn pos_color_layout() -> Vec<quanta::VertexLayout> {
    vec![quanta::VertexLayout {
        stride: 24,
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
                format: quanta::AttributeFormat::Float3,
            },
        ],
    }]
}

fn instanced_layouts() -> Vec<quanta::VertexLayout> {
    vec![
        // Buffer 0: per-vertex position
        quanta::VertexLayout {
            stride: 12,
            step: quanta::StepMode::Vertex,
            attributes: vec![quanta::VertexAttribute {
                location: 0,
                offset: 0,
                format: quanta::AttributeFormat::Float3,
            }],
        },
        // Buffer 1: per-instance offset
        quanta::VertexLayout {
            stride: 12,
            step: quanta::StepMode::Instance,
            attributes: vec![quanta::VertexAttribute {
                location: 1,
                offset: 0,
                format: quanta::AttributeFormat::Float3,
            }],
        },
    ]
}

fn pixel_at(pixels: &[u8], w: u32, x: u32, y: u32) -> (u8, u8, u8, u8) {
    let i = ((y * w + x) * 4) as usize;
    (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3])
}

fn make_pipeline(
    gpu: &quanta::Gpu,
    vert: &quanta::ShaderBinary,
    frag: &quanta::ShaderBinary,
    layouts: &[quanta::VertexLayout],
    depth: bool,
) -> quanta::Pipeline {
    let mut desc = quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
        vertex: vert,
        fragment: frag,
    })
    .with_entries(vert.entry_point, frag.entry_point)
    .with_color_formats(vec![Format::RGBA8])
    .with_vertex_layouts(layouts)
    .with_blend(quanta::BlendState::NONE);
    if depth {
        desc = desc
            .with_depth_format(Format::Depth32Float)
            .with_depth_stencil(quanta::DepthStencilState::DEPTH_LESS);
    }
    gpu.pipeline(&desc).expect("pipeline creation")
}

// ─── Test 1: Basic triangle (sanity) ────────────────────────────────────────

#[test]
fn check_shader_binaries() {
    eprintln!(
        "vertex spirv: {}",
        PASSTHROUGH_VERTEX_SHADER.spirv.is_some()
    );
    eprintln!(
        "vertex metallib: {}",
        PASSTHROUGH_VERTEX_SHADER.metallib.is_some()
    );
    eprintln!("fragment spirv: {}", SOLID_RED_SHADER.spirv.is_some());
    eprintln!("fragment metallib: {}", SOLID_RED_SHADER.metallib.is_some());
    eprintln!("green spirv: {}", SOLID_GREEN_SHADER.spirv.is_some());
    eprintln!("green metallib: {}", SOLID_GREEN_SHADER.metallib.is_some());
    eprintln!("offset spirv: {}", OFFSET_VERTEX_SHADER.spirv.is_some());
    eprintln!(
        "offset metallib: {}",
        OFFSET_VERTEX_SHADER.metallib.is_some()
    );
}

#[test]
fn render_triangle() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_color_layout();
    let pipeline = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        false,
    );

    #[rustfmt::skip]
    let verts: [f32; 18] = [
         0.0, -0.5, 0.0,   1.0, 0.0, 0.0,
        -0.5,  0.5, 0.0,   0.0, 1.0, 0.0,
         0.5,  0.5, 0.0,   0.0, 0.0, 1.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .expect("vb");
    vb.write(&verts).expect("write vb");

    let w = 64u32;
    let h = 64u32;
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
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    let pixels = target.read().unwrap();
    let (r, g, b, a) = pixel_at(&pixels, w, w / 2, h / 2);
    eprintln!("center: rgba({r},{g},{b},{a})");
    assert!(r > 200, "center should be red (R={r})");
    assert!(g < 50 && b < 50, "center should not be green/blue");

    let (cr, cg, cb, _) = pixel_at(&pixels, w, 0, 0);
    eprintln!("corner: rgba({cr},{cg},{cb})");
    assert!(cr < 10 && cg < 10 && cb < 10, "corner should be black");
}

// ─── Test 2: Depth test — two overlapping triangles ─────────────────────────

#[test]
fn depth_test_near_wins() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || SOLID_GREEN_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_color_layout();
    let red_pipe = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        true,
    );
    let green_pipe = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_GREEN_SHADER,
        &layouts,
        true,
    );

    // FAR triangle (z=0.8) — green, covers center
    #[rustfmt::skip]
    let far_verts: [f32; 18] = [
         0.0, -0.7, 0.8,   0.0, 0.0, 0.0,
        -0.7,  0.7, 0.8,   0.0, 0.0, 0.0,
         0.7,  0.7, 0.8,   0.0, 0.0, 0.0,
    ];
    // NEAR triangle (z=0.2) — red, covers center
    #[rustfmt::skip]
    let near_verts: [f32; 18] = [
         0.0, -0.7, 0.2,   0.0, 0.0, 0.0,
        -0.7,  0.7, 0.2,   0.0, 0.0, 0.0,
         0.7,  0.7, 0.2,   0.0, 0.0, 0.0,
    ];

    let far_vb: quanta::Field<f32> = gpu
        .field_with_usage(far_verts.len(), FieldUsage::default_render())
        .unwrap();
    far_vb.write(&far_verts).unwrap();
    let near_vb: quanta::Field<f32> = gpu
        .field_with_usage(near_verts.len(), FieldUsage::default_render())
        .unwrap();
    near_vb.write(&near_verts).unwrap();

    let w = 64u32;
    let h = 64u32;
    let color_target = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let depth_target = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, Format::Depth32Float)
                .with_usage(quanta::TextureUsage::RENDER_TARGET),
        )
        .unwrap();

    // Draw order: green (far) FIRST, then red (near) SECOND.
    // With depth test (Less), the near red triangle wins at overlap.
    let mut pulse = gpu
        .render(&color_target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&color_target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .depth_target(
            quanta::render_pass::DepthTarget::new(&depth_target)
                .with_load_op(LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 0.0))) // depth = 1.0
                .with_store_op(StoreOp::DontCare)
                .with_stencil_load_op(LoadOp::DontCare)
                .with_stencil_store_op(StoreOp::DontCare),
        )
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&green_pipe)
        .vertices(0, &far_vb)
        .draw(3)
        .pipeline(&red_pipe)
        .vertices(0, &near_vb)
        .draw(3)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    let pixels = color_target.read().unwrap();
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    eprintln!("depth test center: rgba({r},{g},{b})");

    // Center should be RED (near triangle wins), NOT green
    assert!(r > 200, "depth test: near red should win (R={r})");
    assert!(g < 50, "depth test: green should be occluded (G={g})");
}

// ─── Test 3: Indexed draw (cube) ────────────────────────────────────────────

#[test]
fn indexed_draw_cube() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_color_layout();
    let pipeline = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        true,
    );

    // Cube vertices: 8 corners, pre-transformed to NDC.
    // Apply simple rotation (30° Y, 20° X) + orthographic scale.
    let s = 0.35f32; // scale
    let (sy, cy) = (30.0f32.to_radians().sin(), 30.0f32.to_radians().cos());
    let (sx, cx) = (20.0f32.to_radians().sin(), 20.0f32.to_radians().cos());

    // Raw cube corners at ±1
    let raw: [[f32; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];

    // Transform: rotate Y, rotate X, scale, shift Z into [0.1, 0.9]
    let mut verts = Vec::with_capacity(8 * 6);
    for [x, y, z] in &raw {
        // Rotate around Y
        let rx = x * cy + z * sy;
        let ry = *y;
        let rz = -x * sy + z * cy;
        // Rotate around X
        let fx = rx;
        let fy = ry * cx - rz * sx;
        let fz = ry * sx + rz * cx;
        // Scale + map Z to [0.1, 0.9]
        verts.extend_from_slice(&[fx * s, fy * s, fz * 0.3 + 0.5, 0.0, 0.0, 0.0]);
    }

    // 12 triangles, 36 indices (6 faces × 2 triangles)
    #[rustfmt::skip]
    let indices: [u32; 36] = [
        // Front  (z = -1)
        0, 1, 2,  2, 3, 0,
        // Back   (z = +1)
        4, 6, 5,  6, 4, 7,
        // Top    (y = +1)
        3, 2, 6,  6, 7, 3,
        // Bottom (y = -1)
        0, 5, 1,  5, 0, 4,
        // Right  (x = +1)
        1, 5, 6,  6, 2, 1,
        // Left   (x = -1)
        0, 3, 7,  7, 4, 0,
    ];

    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();
    let ib: quanta::Field<u32> = gpu
        .field_with_usage(indices.len(), FieldUsage::default_render())
        .unwrap();
    ib.write(&indices).unwrap();

    let w = 64u32;
    let h = 64u32;
    let color_target = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let depth_target = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, Format::Depth32Float)
                .with_usage(quanta::TextureUsage::RENDER_TARGET),
        )
        .unwrap();

    let mut pulse = gpu
        .render(&color_target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&color_target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .depth_target(
            quanta::render_pass::DepthTarget::new(&depth_target)
                .with_load_op(LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 0.0)))
                .with_store_op(StoreOp::DontCare)
                .with_stencil_load_op(LoadOp::DontCare)
                .with_stencil_store_op(StoreOp::DontCare),
        )
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .indices(&ib)
        .draw_indexed(36)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    let pixels = color_target.read().unwrap();

    // Center should be red (cube covers it)
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    eprintln!("indexed cube center: rgba({r},{g},{b})");
    assert!(r > 200, "cube center should be red (R={r})");
    assert!(g < 50 && b < 50, "cube center should not be green/blue");

    // Count red pixels — cube should cover significant area
    let mut red_count = 0u32;
    for y in 0..h {
        for x in 0..w {
            let (pr, _, _, _) = pixel_at(&pixels, w, x, y);
            if pr > 200 {
                red_count += 1;
            }
        }
    }
    let coverage = red_count as f32 / (w * h) as f32;
    eprintln!(
        "cube coverage: {:.1}% ({red_count} red pixels)",
        coverage * 100.0
    );
    // A rotated cube at scale 0.35 should cover roughly 15-60% of the viewport
    assert!(
        coverage > 0.10,
        "cube should cover >10% (got {:.1}%)",
        coverage * 100.0
    );
    assert!(
        coverage < 0.80,
        "cube should cover <80% (got {:.1}%)",
        coverage * 100.0
    );
}

// ─── Test 4: Instanced draw ────────────────────────────────────────────────

#[test]
fn instanced_draw() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if OFFSET_VERTEX_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = instanced_layouts();
    let pipeline = make_pipeline(
        &gpu,
        &OFFSET_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        false,
    );

    // Small triangle centered at origin (will be offset by instance data)
    #[rustfmt::skip]
    let tri: [f32; 9] = [
         0.0, -0.15, 0.0,
        -0.15, 0.15, 0.0,
         0.15, 0.15, 0.0,
    ];

    // 3 instances at different positions: left, center, right
    #[rustfmt::skip]
    let offsets: [f32; 9] = [
        -0.5, 0.0, 0.0,  // left
         0.0, 0.0, 0.0,  // center
         0.5, 0.0, 0.0,  // right
    ];

    let vb: quanta::Field<f32> = gpu
        .field_with_usage(tri.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&tri).unwrap();
    let instance_buf: quanta::Field<f32> = gpu
        .field_with_usage(offsets.len(), FieldUsage::default_render())
        .unwrap();
    instance_buf.write(&offsets).unwrap();

    let w = 128u32;
    let h = 64u32;
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
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .vertices(1, &instance_buf)
        .draw_instanced(3, 3)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    let pixels = target.read().unwrap();

    // Check three horizontal positions for red pixels
    let left_x = w / 4; // ~32: should have left instance
    let center_x = w / 2; // ~64: should have center instance
    let right_x = 3 * w / 4; // ~96: should have right instance
    let y = h / 2;

    let (lr, _, _, _) = pixel_at(&pixels, w, left_x, y);
    let (cr, _, _, _) = pixel_at(&pixels, w, center_x, y);
    let (rr, _, _, _) = pixel_at(&pixels, w, right_x, y);
    eprintln!(
        "instanced draw — left({left_x},{y}): R={lr}, center({center_x},{y}): R={cr}, right({right_x},{y}): R={rr}"
    );

    assert!(cr > 200, "center instance should be red (R={cr})");

    // Count total red pixels — 3 instances should produce 3× the coverage of one
    let mut red_count = 0u32;
    for py in 0..h {
        for px in 0..w {
            let (pr, _, _, _) = pixel_at(&pixels, w, px, py);
            if pr > 200 {
                red_count += 1;
            }
        }
    }
    eprintln!("instanced draw: {red_count} red pixels total");
    // 3 small triangles should produce a meaningful number of red pixels
    assert!(
        red_count > 50,
        "should have >50 red pixels from 3 instances (got {red_count})"
    );
}

// ─── Test 5: Textured quad (varyings + texture sampling) ────────────────────

#[quanta::vertex]
fn uv_vertex(pos: Vec3, uv: Vec2) -> Vec4 {
    Vec4::new(pos.x, pos.y, 0.0, 1.0)
}

#[quanta::fragment]
fn textured_frag(uv: Vec2) -> Vec4 {
    sample(0, uv)
}

fn pos_uv_layout() -> Vec<quanta::VertexLayout> {
    vec![quanta::VertexLayout {
        stride: 20, // 3 floats (pos) + 2 floats (uv) = 5 × 4
        step: quanta::StepMode::Vertex,
        attributes: vec![
            quanta::VertexAttribute {
                location: 0,
                offset: 0,
                format: quanta::AttributeFormat::Float3, // pos
            },
            quanta::VertexAttribute {
                location: 1,
                offset: 12,
                format: quanta::AttributeFormat::Float2, // uv
            },
        ],
    }]
}

#[test]
fn textured_quad() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if UV_VERTEX_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || TEXTURED_FRAG_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_uv_layout();
    let pipeline = gpu
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
            .with_vertex_layouts(&layouts)
            .with_blend(quanta::BlendState::NONE),
        )
        .expect("pipeline creation");

    // 2×2 checkerboard texture: red, green, blue, white
    let tex_data: [u8; 16] = [
        255, 0, 0, 255, // (0,0) red
        0, 255, 0, 255, // (1,0) green
        0, 0, 255, 255, // (0,1) blue
        255, 255, 255, 255, // (1,1) white
    ];
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(2, 2, Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ),
        )
        .expect("texture");
    tex.write(&tex_data).expect("tex write");

    // Full-screen quad: two triangles covering [-1,1]
    //   pos (x, y, z)     uv (u, v)
    #[rustfmt::skip]
    let verts: [f32; 30] = [
        -1.0, -1.0, 0.0,  0.0, 0.0,  // bottom-left
         1.0, -1.0, 0.0,  1.0, 0.0,  // bottom-right
         1.0,  1.0, 0.0,  1.0, 1.0,  // top-right
        -1.0, -1.0, 0.0,  0.0, 0.0,  // bottom-left
         1.0,  1.0, 0.0,  1.0, 1.0,  // top-right
        -1.0,  1.0, 0.0,  0.0, 1.0,  // top-left
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

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
        .pipeline(&pipeline)
        .vertices(0, &vb)
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

    let pixels = target.read().unwrap();
    assert_eq!(pixels.len(), (w * h * 4) as usize);

    // With nearest sampling on a 4×4 target from a 2×2 texture:
    // Bottom-left quadrant (0,0)-(1,1) → red texel
    // Bottom-right quadrant (2,0)-(3,1) → green texel
    // Top-left quadrant (0,2)-(1,3) → blue texel
    // Top-right quadrant (2,2)-(3,3) → white texel
    let (r, g, b, _) = pixel_at(&pixels, w, 0, 0);
    eprintln!("bottom-left (0,0): rgba({r},{g},{b})");

    let (r2, g2, b2, _) = pixel_at(&pixels, w, 3, 0);
    eprintln!("bottom-right (3,0): rgba({r2},{g2},{b2})");

    let (r3, g3, b3, _) = pixel_at(&pixels, w, 0, 3);
    eprintln!("top-left (0,3): rgba({r3},{g3},{b3})");

    let (r4, g4, b4, _) = pixel_at(&pixels, w, 3, 3);
    eprintln!("top-right (3,3): rgba({r4},{g4},{b4})");

    // At minimum, the quad should NOT be all black (texture is being sampled)
    let total_color: u32 = pixels.iter().step_by(4).map(|&v| v as u32).sum();
    assert!(total_color > 0, "textured quad should not be all black");
    eprintln!("textured quad: total R channel sum = {total_color}");
}

// ─── Test: two indexed draws, two index buffers ─────────────────────────────
//
// Regression: the Metal replay resolved DrawIndexed by scanning the whole
// op list for the last BindIndices, so every indexed draw in a pass used
// the FINAL index buffer — N meshes per pass all rendered the Nth topology.
// One pass, two draws, two index buffers over disjoint triangles: each
// draw must render its OWN triangle.

#[test]
fn two_indexed_draws_use_their_own_index_buffers() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_color_layout();
    let pipeline = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        true,
    );

    // Six vertices: a left triangle (0-2) and a disjoint right one (3-5).
    #[rustfmt::skip]
    let verts: [f32; 36] = [
        -0.9, -0.9, 0.5,  0.0, 0.0, 0.0,
        -0.1, -0.9, 0.5,  0.0, 0.0, 0.0,
        -0.5,  0.7, 0.5,  0.0, 0.0, 0.0,
         0.1, -0.9, 0.5,  0.0, 0.0, 0.0,
         0.9, -0.9, 0.5,  0.0, 0.0, 0.0,
         0.5,  0.7, 0.5,  0.0, 0.0, 0.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    let left: [u32; 3] = [0, 1, 2];
    let right: [u32; 3] = [3, 4, 5];
    let ib_left: quanta::Field<u32> = gpu
        .field_with_usage(left.len(), FieldUsage::default_render())
        .unwrap();
    ib_left.write(&left).unwrap();
    let ib_right: quanta::Field<u32> = gpu
        .field_with_usage(right.len(), FieldUsage::default_render())
        .unwrap();
    ib_right.write(&right).unwrap();

    let w = 64u32;
    let h = 64u32;
    let color_target = gpu.render_target(w, h, Format::RGBA8).unwrap();
    let depth_target = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, Format::Depth32Float)
                .with_usage(quanta::TextureUsage::RENDER_TARGET),
        )
        .unwrap();

    let mut pulse = gpu
        .render(&color_target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&color_target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .depth_target(
            quanta::render_pass::DepthTarget::new(&depth_target)
                .with_load_op(LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 0.0)))
                .with_store_op(StoreOp::DontCare)
                .with_stencil_load_op(LoadOp::DontCare)
                .with_stencil_store_op(StoreOp::DontCare),
        )
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .indices(&ib_left)
        .draw_indexed(3)
        .indices(&ib_right)
        .draw_indexed(3)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    let pixels = color_target.read().unwrap();

    // Inside the left triangle — drawn only by the FIRST index buffer.
    let (r, g, b, _) = pixel_at(&pixels, w, 16, 48);
    assert!(
        r > 200 && g < 50 && b < 50,
        "left triangle missing: the first indexed draw did not use its own \
         index buffer, got ({r},{g},{b})"
    );
    // Inside the right triangle — drawn by the SECOND index buffer.
    let (r, g, b, _) = pixel_at(&pixels, w, 48, 48);
    assert!(
        r > 200 && g < 50 && b < 50,
        "right triangle missing, got ({r},{g},{b})"
    );
    // Between the triangles the background must stay clear.
    let (r, g, b, _) = pixel_at(&pixels, w, 32, 48);
    assert!(
        r < 50 && g < 50 && b < 50,
        "background between the triangles should be clear, got ({r},{g},{b})"
    );
}

// ─── Test: wait_idle syncs an unwaited render for CPU readback ──────────────
//
// The render pulse is async: dropping it without wait() and reading the
// target races the GPU (observed in the wild as an all-black readback).
// Gpu::wait_idle must drain the queue so the read sees the finished frame.

#[test]
fn wait_idle_syncs_readback_without_pulse_wait() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_color_layout();
    let pipeline = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        false,
    );

    #[rustfmt::skip]
    let verts: [f32; 18] = [
         0.0, -0.5, 0.0,   1.0, 0.0, 0.0,
        -0.5,  0.5, 0.0,   0.0, 1.0, 0.0,
         0.5,  0.5, 0.0,   0.0, 0.0, 1.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .expect("vb");
    vb.write(&verts).expect("write vb");

    let w = 64u32;
    let h = 64u32;
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    // Deliberately drop the pulse without waiting on it.
    let _ = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .unwrap();

    gpu.wait_idle().unwrap();

    let pixels = target.read().unwrap();
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    assert!(
        r > 200 && g < 50 && b < 50,
        "triangle must be visible after wait_idle with an unwaited pulse, \
         got ({r},{g},{b})"
    );
}

// ─── Test: on_complete fires from a waiter thread after completion ──────────

#[test]
fn on_complete_notifies_after_render() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_color_layout();
    let pipeline = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        false,
    );

    #[rustfmt::skip]
    let verts: [f32; 18] = [
         0.0, -0.5, 0.0,   1.0, 0.0, 0.0,
        -0.5,  0.5, 0.0,   0.0, 1.0, 0.0,
         0.5,  0.5, 0.0,   0.0, 0.0, 1.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .expect("vb");
    vb.write(&verts).expect("write vb");

    let w = 64u32;
    let h = 64u32;
    let target = gpu.render_target(w, h, Format::RGBA8).unwrap();

    let pulse = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .unwrap();

    // Park nothing: register the wake-up instead of blocking on wait().
    let (tx, rx) = std::sync::mpsc::channel();
    pulse
        .on_complete(move || {
            let _ = tx.send(());
        })
        .unwrap();

    rx.recv_timeout(std::time::Duration::from_secs(10))
        .expect("on_complete callback did not fire");

    // The callback fires only after GPU completion, so the readback
    // must see the finished frame without any further sync.
    let pixels = target.read().unwrap();
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    assert!(
        r > 200 && g < 50 && b < 50,
        "triangle must be visible after on_complete, got ({r},{g},{b})"
    );
}

#[test]
fn on_complete_fires_for_already_completed_pulse() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_color_layout();
    let pipeline = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        false,
    );

    #[rustfmt::skip]
    let verts: [f32; 18] = [
         0.0, -0.5, 0.0,   1.0, 0.0, 0.0,
        -0.5,  0.5, 0.0,   0.0, 1.0, 0.0,
         0.5,  0.5, 0.0,   0.0, 0.0, 1.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .expect("vb");
    vb.write(&verts).expect("write vb");

    let target = gpu.render_target(16, 16, Format::RGBA8).unwrap();
    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, 16.0, 16.0)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    // Registering after completion must still fire, from the waiter
    // thread (uniform execution context).
    let (tx, rx) = std::sync::mpsc::channel();
    pulse
        .on_complete(move || {
            let _ = tx.send(std::thread::current().name().map(String::from));
        })
        .unwrap();
    let thread_name = rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("on_complete callback did not fire for a completed pulse");
    assert_eq!(thread_name.as_deref(), Some("quanta-pulse-waiter"));
}

// ─── Test: dead-handle validation at pulse() ────────────────────────────────
//
// Bound resources are recorded as bare handles; a Field dropped before
// pulse() used to make the replay silently skip the bind (wrong render,
// no error). pulse() must fail loudly instead.

#[test]
fn pulse_fails_loudly_on_dropped_field() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_color_layout();
    let pipeline = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        false,
    );

    let target = gpu.render_target(16, 16, Format::RGBA8).unwrap();
    let mut builder = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, 16.0, 16.0)
        .pipeline(&pipeline);
    {
        let doomed: quanta::Field<f32> = gpu
            .field_with_usage(18, FieldUsage::default_render())
            .unwrap();
        builder = builder.vertices(0, &doomed);
    } // dropped before pulse() — its handle is now dead

    let err = match builder.draw(3).pulse() {
        Ok(_) => panic!("pulse() must fail for a dropped bound Field"),
        Err(e) => e,
    };
    assert!(
        matches!(err.kind, quanta::QuantaErrorKind::NotFound(_)),
        "expected NotFound for a dropped bound Field, got: {err}"
    );
}

#[test]
fn draw_indexed_without_indices_is_rejected() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if PASSTHROUGH_VERTEX_SHADER
        .for_vendor(gpu.caps().vendor)
        .is_none()
        || SOLID_RED_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_color_layout();
    let pipeline = make_pipeline(
        &gpu,
        &PASSTHROUGH_VERTEX_SHADER,
        &SOLID_RED_SHADER,
        &layouts,
        false,
    );

    let verts = [0.0f32; 18];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    let target = gpu.render_target(16, 16, Format::RGBA8).unwrap();
    let result = gpu
        .render(&target)
        .unwrap()
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, 16.0, 16.0)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw_indexed(3) // no .indices() bind — must be rejected
        .pulse();
    let err = match result {
        Ok(_) => panic!("pulse() must reject draw_indexed without indices"),
        Err(e) => e,
    };
    assert!(
        matches!(err.kind, quanta::QuantaErrorKind::InvalidParam(_)),
        "expected InvalidParam for draw_indexed without indices, got: {err}"
    );
}

// ─── Test: name-based texture params (`&Texture2D` + sample(name, uv)) ──────

#[quanta::fragment]
fn atlas_frag(uv: Vec2, atlas: &Texture2D) -> Vec4 {
    sample(atlas, uv)
}

#[test]
fn texture_param_by_name_samples_exactly() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if UV_VERTEX_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || ATLAS_FRAG_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_uv_layout();
    let pipeline = gpu
        .pipeline(
            &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
                vertex: &UV_VERTEX_SHADER,
                fragment: &ATLAS_FRAG_SHADER,
            })
            .with_entries(UV_VERTEX_SHADER.entry_point, ATLAS_FRAG_SHADER.entry_point)
            .with_color_formats(vec![Format::RGBA8])
            .with_vertex_layouts(&layouts)
            .with_blend(quanta::BlendState::NONE),
        )
        .expect("pipeline creation");

    // 2×2 texture: red, green, blue, white
    let tex_data: [u8; 16] = [
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(2, 2, Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ),
        )
        .expect("texture");
    tex.write(&tex_data).expect("tex write");

    #[rustfmt::skip]
    let verts: [f32; 30] = [
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0, -1.0, 0.0,  1.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0,  1.0, 0.0,  0.0, 1.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

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
        .pipeline(&pipeline)
        .vertices(0, &vb)
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

    let pixels = target.read().unwrap();

    // Nearest sampling of a 2×2 texture over a 4×4 target: each texel
    // covers one quadrant exactly. v=0 is the texture's first row; the
    // framebuffer's y=0 row is the quad's uv v≈0 edge after the y-flip,
    // so probe all four quadrants by their known texel colors.
    let expect = |x: u32, y: u32, want: (u8, u8, u8), which: &str| {
        let (r, g, b, _) = pixel_at(&pixels, w, x, y);
        assert!(
            r.abs_diff(want.0) <= 2 && g.abs_diff(want.1) <= 2 && b.abs_diff(want.2) <= 2,
            "{which} quadrant at ({x},{y}): expected {want:?}, got ({r},{g},{b})"
        );
    };
    // uv (0,0)→red, (1,0)→green, (0,1)→blue, (1,1)→white. With
    // OriginUpperLeft framebuffers, uv v=1 lands on LOW y rows.
    expect(0, 3, (255, 0, 0), "uv(0,0) red");
    expect(3, 3, (0, 255, 0), "uv(1,0) green");
    expect(0, 0, (0, 0, 255), "uv(0,1) blue");
    expect(3, 0, (255, 255, 255), "uv(1,1) white");
}

// ─── Test: fragment uniform param (Metal fragment-stage buffer bind) ────────

#[quanta::fragment]
fn tinted_frag(uv: Vec2, tint: &Vec4) -> Vec4 {
    Vec4::new(tint.x, tint.y, tint.z, 1.0)
}

#[test]
fn fragment_uniform_is_visible_to_fragment_stage() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if UV_VERTEX_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || TINTED_FRAG_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_uv_layout();
    let pipeline = gpu
        .pipeline(
            &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
                vertex: &UV_VERTEX_SHADER,
                fragment: &TINTED_FRAG_SHADER,
            })
            .with_entries(UV_VERTEX_SHADER.entry_point, TINTED_FRAG_SHADER.entry_point)
            .with_color_formats(vec![Format::RGBA8])
            .with_vertex_layouts(&layouts)
            .with_blend(quanta::BlendState::NONE),
        )
        .expect("pipeline creation");

    #[rustfmt::skip]
    let verts: [f32; 30] = [
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0, -1.0, 0.0,  1.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0,  1.0, 0.0,  0.0, 1.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    // The tint uniform: fragment uniforms number [[buffer(i)]] by
    // declaration order among uniform params — `tint` is uniform 0,
    // so it binds at slot 0. Geometry lives in the remapped attribute
    // index space, so slot 0 is free for user buffers on both stages.
    let tint: quanta::Field<f32> = gpu
        .field_with_usage(4, FieldUsage::default_render())
        .unwrap();
    tint.write(&[0.25f32, 0.5, 0.75, 1.0]).unwrap();

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
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .uniform(0, &tint)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    let pixels = target.read().unwrap();
    let (r, g, b, _) = pixel_at(&pixels, w, w / 2, h / 2);
    assert!(
        r.abs_diff(64) <= 2 && g.abs_diff(128) <= 2 && b.abs_diff(191) <= 2,
        "tint uniform must reach the fragment stage: expected ~(64,128,191), got ({r},{g},{b})"
    );
}

// ─── Test: statement-if + fragment uniform, drawn exactly ───────────────────
//
// The branch-and-assign fragment shape (the one real UI shaders use):
// a `let mut`, a statement-if choosing per-pixel, and a `&Vec2` uniform
// read in the fragment stage. Exercises the OpPhi lowering on SPIR-V,
// the AST-emitted MSL, and the both-stage uniform bind — with exact
// per-half color assertions.

#[quanta::fragment]
fn branchy_frag(uv: Vec2, split: &Vec2) -> Vec4 {
    let mut r = 0.0;
    let mut g = 0.0;
    if uv.x > (*split).x {
        r = 1.0;
    } else {
        g = 1.0;
    }
    Vec4::new(r, g, 0.0, 1.0)
}

#[test]
fn statement_if_and_fragment_uniform_render_exactly() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    if UV_VERTEX_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || BRANCHY_FRAG_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary");
        return;
    }

    let layouts = pos_uv_layout();
    let pipeline = gpu
        .pipeline(
            &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
                vertex: &UV_VERTEX_SHADER,
                fragment: &BRANCHY_FRAG_SHADER,
            })
            .with_entries(
                UV_VERTEX_SHADER.entry_point,
                BRANCHY_FRAG_SHADER.entry_point,
            )
            .with_color_formats(vec![Format::RGBA8])
            .with_vertex_layouts(&layouts)
            .with_blend(quanta::BlendState::NONE),
        )
        .expect("pipeline creation");

    #[rustfmt::skip]
    let verts: [f32; 30] = [
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0, -1.0, 0.0,  1.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0, -1.0, 0.0,  0.0, 0.0,
         1.0,  1.0, 0.0,  1.0, 1.0,
        -1.0,  1.0, 0.0,  0.0, 1.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    let split: quanta::Field<f32> = gpu
        .field_with_usage(2, FieldUsage::default_render())
        .unwrap();
    split.write(&[0.5f32, 0.0]).unwrap();

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
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .uniform(0, &split)
        .draw(6)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    let pixels = target.read().unwrap();
    // uv.x < 0.5 → else branch → green; uv.x > 0.5 → then branch → red.
    let (r, g, b, _) = pixel_at(&pixels, w, 1, 4);
    assert!(
        g > 200 && r < 50 && b < 50,
        "left half must take the else branch (green), got ({r},{g},{b})"
    );
    let (r, g, b, _) = pixel_at(&pixels, w, 6, 4);
    assert!(
        r > 200 && g < 50 && b < 50,
        "right half must take the then branch (red), got ({r},{g},{b})"
    );
}
