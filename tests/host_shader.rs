//! Tier 1 (host, no GPU) conformance tests — shader compilation output (proc macros).
//!
//! Tests that all macro types produce valid output:
//! - #[quanta::kernel] produces KernelBinary with msl + wgsl
//! - #[quanta::vertex] produces ShaderBinary with msl + wgsl
//! - #[quanta::fragment] produces ShaderBinary with msl + wgsl
//! - #[quanta::device] produces __QUANTA_DEVICE_* constant
//! - #[quanta::gpu_type] produces GPU_SIZE, GPU_FIELDS, GpuType impl, MSL/WGSL strings
//!
//! Run: cargo test --test host_shader

// ===========================================================================
// #[quanta::kernel] — produces KernelBinary with msl + wgsl
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
fn kernel_test_add_produces_binary() {
    assert!(
        TEST_ADD_BINARY.msl.is_some(),
        "kernel must produce MSL output"
    );
    assert!(
        TEST_ADD_BINARY.wgsl.is_some(),
        "kernel must produce WGSL output"
    );
}

#[test]
fn kernel_test_scale_produces_binary() {
    assert!(TEST_SCALE_BINARY.msl.is_some());
    assert!(TEST_SCALE_BINARY.wgsl.is_some());
}

#[test]
fn kernel_opt_o0_produces_binary() {
    assert!(TEST_IDENTITY_BINARY.msl.is_some());
    assert!(TEST_IDENTITY_BINARY.wgsl.is_some());
}

#[test]
fn kernel_binary_msl_is_valid_metal() {
    let msl = TEST_ADD_BINARY.msl.unwrap();
    assert!(
        msl.contains("kernel void test_add"),
        "MSL must have kernel qualifier: {}",
        msl
    );
    assert!(
        msl.contains("thread_position_in_grid"),
        "MSL must use thread_position_in_grid: {}",
        msl
    );
    assert!(
        msl.contains("device"),
        "MSL must have device pointer qualifier: {}",
        msl
    );
}

#[test]
fn kernel_binary_wgsl_is_valid_wgsl() {
    let wgsl = TEST_ADD_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("@compute"),
        "WGSL must have @compute decorator: {}",
        wgsl
    );
    assert!(
        wgsl.contains("fn test_add"),
        "WGSL must have fn test_add: {}",
        wgsl
    );
    assert!(
        wgsl.contains("global_invocation_id"),
        "WGSL must use global_invocation_id: {}",
        wgsl
    );
}

#[test]
fn kernel_binary_for_vendor_apple() {
    let binary = TEST_ADD_BINARY.for_vendor(quanta::Vendor::Apple);
    assert!(binary.is_some(), "Apple vendor must get a kernel binary");
    let text = core::str::from_utf8(binary.unwrap()).unwrap();
    assert!(text.contains("kernel void"));
}

#[test]
fn kernel_binary_for_vendor_unknown() {
    let binary = TEST_ADD_BINARY.for_vendor(quanta::Vendor::Unknown);
    assert!(
        binary.is_some(),
        "Unknown vendor should fall back to wgsl/spirv"
    );
}

// ===========================================================================
// #[quanta::vertex] — produces ShaderBinary with msl + wgsl
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
    assert!(shader.msl.is_some(), "vertex must produce MSL");
    assert!(shader.wgsl.is_some(), "vertex must produce WGSL");
    assert_eq!(shader.entry_point, "vert_passthrough");
    assert_eq!(shader.stage, quanta::ShaderStage::Vertex);
}

#[test]
fn vertex_static_exists() {
    assert!(VERT_PASSTHROUGH_SHADER.msl.is_some());
    assert!(VERT_PASSTHROUGH_SHADER.wgsl.is_some());
    assert_eq!(VERT_PASSTHROUGH_SHADER.stage, quanta::ShaderStage::Vertex);
}

#[test]
fn vertex_with_uniform_produces_shader() {
    let shader = vert_with_uniform();
    assert!(shader.msl.is_some());
    assert!(shader.wgsl.is_some());
    assert_eq!(shader.entry_point, "vert_with_uniform");
}

// ===========================================================================
// #[quanta::fragment] — produces ShaderBinary with msl + wgsl
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
    assert!(shader.msl.is_some(), "fragment must produce MSL");
    assert!(shader.wgsl.is_some(), "fragment must produce WGSL");
    assert_eq!(shader.entry_point, "frag_red");
    assert_eq!(shader.stage, quanta::ShaderStage::Fragment);
}

#[test]
fn fragment_static_exists() {
    assert!(FRAG_RED_SHADER.msl.is_some());
    assert!(FRAG_RED_SHADER.wgsl.is_some());
    assert_eq!(FRAG_RED_SHADER.stage, quanta::ShaderStage::Fragment);
}

#[test]
fn fragment_color_shader() {
    let shader = frag_color();
    assert!(shader.msl.is_some());
    assert!(shader.wgsl.is_some());
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
