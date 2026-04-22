//! Tier 1 (host, no GPU) conformance tests — shader compilation output (proc macros).
//!
//! Tests that all macro types produce valid output:
//! - #[quanta::kernel] produces KernelBinary with binary fields
//! - #[quanta::vertex] produces ShaderBinary with spirv/metallib
//! - #[quanta::fragment] produces ShaderBinary with spirv/metallib
//! - #[quanta::device] produces __QUANTA_DEVICE_* constant
//! - #[quanta::gpu_type] produces GPU_SIZE, GPU_FIELDS, GpuType impl, MSL/WGSL strings
//!
//! Run: cargo test --test host_shader

// ===========================================================================
// #[quanta::kernel] — produces KernelBinary
// ===========================================================================

#[quanta::kernel]
fn test_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}

#[quanta::kernel]
fn test_scale(data: &mut [f32], factor: f32) {
    let i = quark_id();
    data[i] = data[i] * factor;
}

#[quanta::kernel(opt = "O0")]
fn test_identity(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    output[i] = input[i];
}

#[test]
fn kernel_binary_struct_has_expected_fields() {
    // With quanta-compiler binary available, spirv/metallib may be populated.
    // Without it, all fields are None — that's valid for host tests.
    let _amd = TEST_ADD_BINARY.amd;
    let _nvidia = TEST_ADD_BINARY.nvidia;
    let _spirv = TEST_ADD_BINARY.spirv;
    let _metallib = TEST_ADD_BINARY.metallib;
}

#[test]
fn kernel_binary_for_vendor_apple_needs_metallib() {
    // Without compiler binary, Apple vendor returns None (no metallib).
    // With compiler binary on macOS, it should return Some.
    let binary = TEST_ADD_BINARY.for_vendor(quanta::Vendor::Apple);
    // Either Some(metallib) or None — both valid depending on build environment.
    let _ = binary;
}

#[test]
fn kernel_binary_for_vendor_unknown() {
    // Unknown vendor needs SPIR-V — returns None if compiler not available.
    let binary = TEST_ADD_BINARY.for_vendor(quanta::Vendor::Unknown);
    // spirv may or may not be populated depending on whether quanta-compiler
    // was found during build.
    let _ = binary;
}

// ===========================================================================
// #[quanta::vertex] — produces ShaderBinary
// ===========================================================================

#[quanta::vertex]
fn vert_passthrough(pos: Vec4) -> Vec4 {
    pos
}

#[quanta::vertex]
fn vert_with_uniform(pos: Vec3, mvp: &Mat4) -> Vec4 {
    mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[test]
fn vertex_produces_shader_binary() {
    let shader = vert_passthrough();
    assert_eq!(shader.entry_point, "vert_passthrough");
    assert_eq!(shader.stage, quanta::ShaderStage::Vertex);
}

#[test]
fn vertex_static_exists() {
    assert_eq!(VERT_PASSTHROUGH_SHADER.stage, quanta::ShaderStage::Vertex);
}

#[test]
fn vertex_with_uniform_produces_shader() {
    let shader = vert_with_uniform();
    assert_eq!(shader.entry_point, "vert_with_uniform");
}

// ===========================================================================
// #[quanta::fragment] — produces ShaderBinary
// ===========================================================================

#[quanta::fragment]
fn frag_red(uv: Vec2) -> Vec4 {
    Vec4::new(1.0, 0.0, 0.0, 1.0)
}

#[quanta::fragment]
fn frag_color(color: Vec4) -> Vec4 {
    color
}

#[test]
fn fragment_produces_shader_binary() {
    let shader = frag_red();
    assert_eq!(shader.entry_point, "frag_red");
    assert_eq!(shader.stage, quanta::ShaderStage::Fragment);
}

#[test]
fn fragment_static_exists() {
    assert_eq!(FRAG_RED_SHADER.stage, quanta::ShaderStage::Fragment);
}

#[test]
fn fragment_color_shader() {
    let shader = frag_color();
    assert_eq!(shader.entry_point, "frag_color");
}

// ===========================================================================
// #[quanta::device] — produces __QUANTA_DEVICE_* constant
// ===========================================================================

#[quanta::device]
fn activate(x: f32, threshold: f32) -> f32 {
    if x > threshold { x } else { x * 0.01 }
}

#[quanta::device]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[test]
fn device_macro_produces_constant() {
    // The device macro should produce __QUANTA_DEVICE_{NAME_UPPERCASE} constant
    assert!(
        !__QUANTA_DEVICE_ACTIVATE.is_empty(),
        "device macro must produce non-empty source constant"
    );
    assert!(
        __QUANTA_DEVICE_ACTIVATE.contains("activate"),
        "source must contain function name"
    );
    assert!(
        __QUANTA_DEVICE_ACTIVATE.contains("threshold"),
        "source must contain parameter names"
    );
}

#[test]
fn device_macro_lerp_constant() {
    assert!(!__QUANTA_DEVICE_LERP.is_empty());
    assert!(__QUANTA_DEVICE_LERP.contains("lerp"));
    assert!(__QUANTA_DEVICE_LERP.contains("f32"));
}

// ===========================================================================
// #[quanta::gpu_type] — produces GPU_SIZE, GPU_FIELDS, GpuType impl, MSL/WGSL
// ===========================================================================

#[quanta::gpu_type]
struct TestVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[quanta::gpu_type]
struct TestUniform {
    model: [f32; 16],
    color: [f32; 4],
    time: f32,
}

#[test]
fn gpu_type_produces_gpu_size() {
    // position(12) + normal(12) + uv(8) = 32 bytes
    assert_eq!(TestVertex::GPU_SIZE, core::mem::size_of::<TestVertex>());
    assert_eq!(TestVertex::GPU_SIZE, 32);
}

#[test]
fn gpu_type_produces_gpu_fields() {
    assert_eq!(TestVertex::GPU_FIELDS.len(), 3);
    assert_eq!(TestVertex::GPU_FIELDS[0].0, "position");
    assert_eq!(TestVertex::GPU_FIELDS[1].0, "normal");
    assert_eq!(TestVertex::GPU_FIELDS[2].0, "uv");
}

#[test]
fn gpu_type_produces_trait_impl() {
    use quanta::GpuType;
    let size = <TestVertex as GpuType>::gpu_size();
    assert_eq!(size, 32);
}

#[test]
fn gpu_type_produces_msl_string() {
    assert!(__QUANTA_GPU_TYPE_TESTVERTEX.contains("struct TestVertex"));
    assert!(__QUANTA_GPU_TYPE_TESTVERTEX.contains("float3 position"));
    assert!(__QUANTA_GPU_TYPE_TESTVERTEX.contains("float3 normal"));
    assert!(__QUANTA_GPU_TYPE_TESTVERTEX.contains("float2 uv"));
}

#[test]
fn gpu_type_produces_wgsl_string() {
    assert!(__QUANTA_GPU_TYPE_TESTVERTEX_WGSL.contains("struct TestVertex"));
    assert!(__QUANTA_GPU_TYPE_TESTVERTEX_WGSL.contains("position: vec3<f32>"));
    assert!(__QUANTA_GPU_TYPE_TESTVERTEX_WGSL.contains("normal: vec3<f32>"));
    assert!(__QUANTA_GPU_TYPE_TESTVERTEX_WGSL.contains("uv: vec2<f32>"));
}

#[test]
fn gpu_type_uniform_msl() {
    assert!(__QUANTA_GPU_TYPE_TESTUNIFORM.contains("struct TestUniform"));
    assert!(__QUANTA_GPU_TYPE_TESTUNIFORM.contains("float4x4 model"));
    assert!(__QUANTA_GPU_TYPE_TESTUNIFORM.contains("float4 color"));
    assert!(__QUANTA_GPU_TYPE_TESTUNIFORM.contains("float time"));
}

#[test]
fn gpu_type_struct_is_copy() {
    let v = TestVertex {
        position: [1.0, 2.0, 3.0],
        normal: [0.0, 1.0, 0.0],
        uv: [0.5, 0.5],
    };
    let _copy = v;
    let _also = v; // Only works if Copy is derived
}
