//! Compile-time verification that #[quanta::vertex] and #[quanta::fragment]
//! macros produce valid ShaderBinary statics.
//!
//! Run: cargo test --test shader_compile_test

// === Vertex shaders ===

#[quanta::vertex]
fn simple_passthrough(pos: Vec4) -> Vec4 {
    pos
}

#[quanta::vertex]
fn transform_mvp(pos: Vec3, normal: Vec3, mvp: &Mat4) -> Vec4 {
    mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::vertex]
fn pos_only(pos: Vec2) -> Vec4 {
    Vec4::new(pos.x, pos.y, 0.0, 1.0)
}

// === Fragment shaders ===

#[quanta::fragment]
fn solid_red(uv: Vec2) -> Vec4 {
    Vec4::new(1.0, 0.0, 0.0, 1.0)
}

#[quanta::fragment]
fn passthrough_color(color: Vec4) -> Vec4 {
    color
}

#[quanta::fragment]
fn shade_uv(uv: Vec2, color: Vec4) -> Vec4 {
    Vec4::new(uv.x, uv.y, color.z, 1.0)
}

// === Tests ===

#[test]
fn vertex_simple_passthrough_compiles() {
    let shader = simple_passthrough();
    assert_eq!(shader.entry_point, "simple_passthrough");
    assert_eq!(shader.stage, quanta::ShaderStage::Vertex);
}

#[test]
fn vertex_transform_mvp_compiles() {
    let shader = transform_mvp();
    assert_eq!(shader.entry_point, "transform_mvp");
    assert_eq!(shader.stage, quanta::ShaderStage::Vertex);
}

#[test]
fn vertex_static_exists() {
    assert_eq!(SIMPLE_PASSTHROUGH_SHADER.stage, quanta::ShaderStage::Vertex);
    assert_eq!(TRANSFORM_MVP_SHADER.stage, quanta::ShaderStage::Vertex);
}

#[test]
fn fragment_solid_red_compiles() {
    let shader = solid_red();
    assert_eq!(shader.entry_point, "solid_red");
    assert_eq!(shader.stage, quanta::ShaderStage::Fragment);
}

#[test]
fn fragment_shade_uv_compiles() {
    let shader = shade_uv();
    assert_eq!(shader.entry_point, "shade_uv");
    assert_eq!(shader.stage, quanta::ShaderStage::Fragment);
}

#[test]
fn fragment_static_exists() {
    assert_eq!(SOLID_RED_SHADER.stage, quanta::ShaderStage::Fragment);
    assert_eq!(SHADE_UV_SHADER.stage, quanta::ShaderStage::Fragment);
}

#[test]
fn shader_for_vendor_apple_returns_binary_or_none() {
    let shader = simple_passthrough();
    // Without binary compilation, for_vendor returns None.
    // With metallib, it returns Some.
    let _binary = shader.for_vendor(quanta::Vendor::Apple);
}

#[test]
fn shader_for_vendor_nvidia_returns_spirv_or_none() {
    let shader = simple_passthrough();
    // Without binary compilation, for_vendor returns None.
    // With SPIR-V, it returns Some.
    let _binary = shader.for_vendor(quanta::Vendor::Nvidia);
}

#[test]
fn pos_only_vertex_2d() {
    let shader = pos_only();
    assert_eq!(shader.entry_point, "pos_only");
    assert_eq!(shader.stage, quanta::ShaderStage::Vertex);
}
