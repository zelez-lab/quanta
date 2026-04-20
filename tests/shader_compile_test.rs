//! Compile-time verification that #[quanta::vertex] and #[quanta::fragment]
//! macros produce valid ShaderBinary statics with MSL + WGSL source.
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
fn vertex_simple_passthrough_has_msl_and_wgsl() {
    let shader = simple_passthrough();
    assert!(shader.msl.is_some(), "vertex MSL must be generated");
    assert!(shader.wgsl.is_some(), "vertex WGSL must be generated");
    assert_eq!(shader.entry_point, "simple_passthrough");
    assert_eq!(shader.stage, quanta::ShaderStage::Vertex);
}

#[test]
fn vertex_transform_mvp_has_msl_and_wgsl() {
    let shader = transform_mvp();
    assert!(shader.msl.is_some());
    assert!(shader.wgsl.is_some());
    assert_eq!(shader.entry_point, "transform_mvp");
    assert_eq!(shader.stage, quanta::ShaderStage::Vertex);
}

#[test]
fn vertex_msl_has_vertex_qualifier() {
    let msl = SIMPLE_PASSTHROUGH_SHADER.msl.unwrap();
    assert!(
        msl.contains("vertex float4 simple_passthrough"),
        "MSL must contain vertex qualifier: {}",
        msl
    );
}

#[test]
fn vertex_msl_has_attribute_annotations() {
    let msl = TRANSFORM_MVP_SHADER.msl.unwrap();
    assert!(
        msl.contains("[[attribute(0)]]"),
        "MSL must annotate first attribute: {}",
        msl
    );
    assert!(
        msl.contains("[[attribute(1)]]"),
        "MSL must annotate second attribute: {}",
        msl
    );
    assert!(
        msl.contains("[[buffer(0)]]"),
        "MSL must annotate uniform buffer: {}",
        msl
    );
}

#[test]
fn vertex_wgsl_has_vertex_decorator() {
    let wgsl = SIMPLE_PASSTHROUGH_SHADER.wgsl.unwrap();
    assert!(
        wgsl.contains("@vertex"),
        "WGSL must contain @vertex decorator: {}",
        wgsl
    );
}

#[test]
fn vertex_wgsl_has_input_struct() {
    let wgsl = TRANSFORM_MVP_SHADER.wgsl.unwrap();
    assert!(
        wgsl.contains("struct VertexInput"),
        "WGSL must define VertexInput struct: {}",
        wgsl
    );
    assert!(
        wgsl.contains("@location(0)"),
        "WGSL must have location annotations: {}",
        wgsl
    );
}

#[test]
fn vertex_wgsl_has_uniform_binding() {
    let wgsl = TRANSFORM_MVP_SHADER.wgsl.unwrap();
    assert!(
        wgsl.contains("var<uniform> mvp"),
        "WGSL must have uniform binding for mvp: {}",
        wgsl
    );
}

#[test]
fn fragment_solid_red_has_msl_and_wgsl() {
    let shader = solid_red();
    assert!(shader.msl.is_some(), "fragment MSL must be generated");
    assert!(shader.wgsl.is_some(), "fragment WGSL must be generated");
    assert_eq!(shader.entry_point, "solid_red");
    assert_eq!(shader.stage, quanta::ShaderStage::Fragment);
}

#[test]
fn fragment_msl_has_fragment_qualifier() {
    let msl = SOLID_RED_SHADER.msl.unwrap();
    assert!(
        msl.contains("fragment float4 solid_red"),
        "MSL must contain fragment qualifier: {}",
        msl
    );
}

#[test]
fn fragment_msl_has_stage_in() {
    let msl = SHADE_UV_SHADER.msl.unwrap();
    assert!(
        msl.contains("[[stage_in]]"),
        "MSL fragment must use stage_in for interpolated inputs: {}",
        msl
    );
}

#[test]
fn fragment_wgsl_has_fragment_decorator() {
    let wgsl = SOLID_RED_SHADER.wgsl.unwrap();
    assert!(
        wgsl.contains("@fragment"),
        "WGSL must contain @fragment decorator: {}",
        wgsl
    );
}

#[test]
fn fragment_wgsl_has_input_struct() {
    let wgsl = SHADE_UV_SHADER.wgsl.unwrap();
    assert!(
        wgsl.contains("struct FragmentInput"),
        "WGSL must define FragmentInput struct: {}",
        wgsl
    );
}

#[test]
fn fragment_wgsl_has_output_location() {
    let wgsl = SOLID_RED_SHADER.wgsl.unwrap();
    assert!(
        wgsl.contains("@location(0)"),
        "WGSL fragment must have output location: {}",
        wgsl
    );
}

#[test]
fn shader_for_vendor_apple_returns_msl() {
    let shader = simple_passthrough();
    let binary = shader.for_vendor(quanta::Vendor::Apple);
    assert!(binary.is_some(), "Apple vendor should get MSL shader");
    let text = core::str::from_utf8(binary.unwrap()).unwrap();
    assert!(
        text.contains("vertex"),
        "Apple binary should be MSL text: {}",
        text
    );
}

#[test]
fn shader_for_vendor_nvidia_returns_wgsl() {
    let shader = simple_passthrough();
    let binary = shader.for_vendor(quanta::Vendor::Nvidia);
    assert!(binary.is_some(), "NVIDIA vendor should get WGSL fallback");
    let text = core::str::from_utf8(binary.unwrap()).unwrap();
    assert!(
        text.contains("@vertex"),
        "NVIDIA binary should be WGSL text: {}",
        text
    );
}

#[test]
fn pos_only_vertex_2d() {
    let shader = pos_only();
    assert_eq!(shader.entry_point, "pos_only");
    let msl = shader.msl.unwrap();
    assert!(
        msl.contains("float2 pos"),
        "MSL should use float2 for Vec2: {}",
        msl
    );
}
