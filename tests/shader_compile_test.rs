#![cfg(feature = "render")]
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

// === Phase 2: binary verification ===

#[test]
fn vertex_spirv_is_populated() {
    let shader = simple_passthrough();
    assert!(
        shader.spirv.is_some(),
        "vertex shader must produce SPIR-V binary",
    );
    let spirv = shader.spirv.unwrap();
    assert!(
        spirv.len() >= 20,
        "SPIR-V binary too small: {} bytes",
        spirv.len()
    );
    // Verify SPIR-V magic number
    let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
    assert_eq!(magic, 0x07230203, "bad SPIR-V magic: 0x{:08x}", magic);
}

#[test]
fn fragment_spirv_is_populated() {
    let shader = solid_red();
    assert!(
        shader.spirv.is_some(),
        "fragment shader must produce SPIR-V binary",
    );
    let spirv = shader.spirv.unwrap();
    assert!(
        spirv.len() >= 20,
        "SPIR-V binary too small: {} bytes",
        spirv.len()
    );
    let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
    assert_eq!(magic, 0x07230203, "bad SPIR-V magic: 0x{:08x}", magic);
}

#[test]
fn vertex_metallib_is_populated_on_macos() {
    let shader = simple_passthrough();
    // metallib requires xcrun (macOS only)
    if cfg!(target_os = "macos") {
        assert!(
            shader.metallib.is_some(),
            "vertex shader must produce metallib on macOS",
        );
        let metallib = shader.metallib.unwrap();
        assert!(metallib.len() >= 4, "metallib too small");
        // Metal library magic: "MTLB"
        assert_eq!(&metallib[..4], b"MTLB", "bad metallib magic");
    }
}

#[test]
fn fragment_metallib_is_populated_on_macos() {
    let shader = solid_red();
    if cfg!(target_os = "macos") {
        assert!(
            shader.metallib.is_some(),
            "fragment shader must produce metallib on macOS",
        );
        let metallib = shader.metallib.unwrap();
        assert!(metallib.len() >= 4, "metallib too small");
        assert_eq!(&metallib[..4], b"MTLB", "bad metallib magic");
    }
}

#[test]
fn vertex_with_multiple_inputs_spirv() {
    let shader = transform_mvp();
    assert!(
        shader.spirv.is_some(),
        "vertex shader with uniform must produce SPIR-V",
    );
}

#[test]
fn fragment_with_multiple_inputs_spirv() {
    let shader = shade_uv();
    assert!(
        shader.spirv.is_some(),
        "fragment shader with multiple inputs must produce SPIR-V",
    );
}

#[test]
fn vertex_for_vendor_returns_binary() {
    let shader = simple_passthrough();
    // Nvidia/AMD/Intel use SPIR-V
    assert!(
        shader.for_vendor(quanta::Vendor::Nvidia).is_some(),
        "vertex for_vendor(Nvidia) must return SPIR-V binary",
    );
}
