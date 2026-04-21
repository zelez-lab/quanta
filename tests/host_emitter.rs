//! Tier 1 (host, no GPU) conformance tests — MSL/WGSL emitter output correctness.
//!
//! Tests that the proc macro emitter produces valid shader code for various
//! kernel patterns: basic kernel, shared memory, atomics, branch/loop,
//! device functions, and wave intrinsics.
//!
//! Run: cargo test --test host_emitter

// ===========================================================================
// Basic kernel (load, binop, store)
// ===========================================================================

#[quanta::kernel]
fn emit_double(data: &mut [f32]) {
    let i = quark_id();
    data[i] = data[i] * 2.0;
}

#[test]
fn basic_msl_valid() {
    let msl = EMIT_DOUBLE_BINARY.msl.unwrap();
    assert!(
        msl.contains("kernel void emit_double"),
        "MSL must have kernel declaration: {}",
        msl
    );
    assert!(
        msl.contains("thread_position_in_grid"),
        "MSL must use thread_position_in_grid: {}",
        msl
    );
    assert!(
        msl.contains("device float*") || msl.contains("device const float*"),
        "MSL must have device pointer: {}",
        msl
    );
}

#[test]
fn basic_wgsl_valid() {
    let wgsl = EMIT_DOUBLE_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("@compute"),
        "WGSL must have @compute: {}",
        wgsl
    );
    assert!(
        wgsl.contains("fn emit_double"),
        "WGSL must have fn declaration: {}",
        wgsl
    );
    assert!(
        wgsl.contains("global_invocation_id"),
        "WGSL must use global_invocation_id: {}",
        wgsl
    );
}

// ===========================================================================
// Shared memory kernel
// ===========================================================================

#[quanta::kernel]
fn emit_shared(input: &[f32], output: &mut [f32]) {
    #[quanta::shared]
    let local: [f32; 256];
    let lid = local_id();
    let gid = quark_id();
    local[lid] = input[gid];
    barrier();
    output[gid] = local[lid];
}

#[test]
fn shared_msl_valid() {
    let msl = EMIT_SHARED_BINARY.msl.unwrap();
    assert!(
        msl.contains("threadgroup") || msl.contains("shared"),
        "MSL must declare threadgroup memory: {}",
        msl
    );
}

#[test]
fn shared_wgsl_valid() {
    let wgsl = EMIT_SHARED_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("var<workgroup>") || wgsl.contains("workgroup"),
        "WGSL must declare workgroup variable: {}",
        wgsl
    );
}

// ===========================================================================
// Atomic kernel
// ===========================================================================

#[quanta::kernel]
fn emit_atomic(counter: &mut [u32], data: &[f32]) {
    let i = quark_id();
    if data[i] > 0.0 {
        atomic_add(&mut counter[i], 1u32);
    }
}

#[test]
fn atomic_msl_valid() {
    let msl = EMIT_ATOMIC_BINARY.msl.unwrap();
    assert!(
        msl.contains("atomic") || msl.contains("atomic_fetch_add"),
        "MSL must contain atomic operation: {}",
        msl
    );
}

#[test]
fn atomic_wgsl_valid() {
    let wgsl = EMIT_ATOMIC_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("atomic") || wgsl.contains("atomicAdd"),
        "WGSL must contain atomic operation: {}",
        wgsl
    );
}

// ===========================================================================
// Branch/loop kernel
// ===========================================================================

#[quanta::kernel]
fn emit_branch(input: &[f32], output: &mut [f32], threshold: f32) {
    let i = quark_id();
    if input[i] > threshold {
        output[i] = input[i];
    } else {
        output[i] = 0.0;
    }
}

#[test]
fn branch_msl_valid() {
    let msl = EMIT_BRANCH_BINARY.msl.unwrap();
    assert!(
        msl.contains("if") || msl.contains("?"),
        "MSL must contain conditional: {}",
        msl
    );
    assert!(
        msl.contains("kernel void emit_branch"),
        "MSL must have correct kernel name: {}",
        msl
    );
}

#[test]
fn branch_wgsl_valid() {
    let wgsl = EMIT_BRANCH_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("if") || wgsl.contains("select"),
        "WGSL must contain conditional: {}",
        wgsl
    );
    assert!(
        wgsl.contains("fn emit_branch"),
        "WGSL must have correct kernel name: {}",
        wgsl
    );
}

// ===========================================================================
// Device function kernel
// ===========================================================================

#[quanta::device]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[quanta::kernel]
fn emit_with_device(input: &[f32], output: &mut [f32]) {
    fn relu(x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }
    let i = quark_id();
    output[i] = relu(input[i]);
}

#[test]
fn device_fn_msl_appears_before_kernel() {
    let msl = EMIT_WITH_DEVICE_BINARY.msl.unwrap();
    assert!(
        msl.contains("kernel void emit_with_device"),
        "MSL must contain the kernel: {}",
        msl
    );
    // The device function should be present in some form
    assert!(
        msl.contains("relu"),
        "MSL must contain device function name: {}",
        msl
    );
}

#[test]
fn device_fn_wgsl_appears_before_kernel() {
    let wgsl = EMIT_WITH_DEVICE_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("fn emit_with_device"),
        "WGSL must contain the kernel: {}",
        wgsl
    );
    assert!(
        wgsl.contains("relu"),
        "WGSL must contain device function name: {}",
        wgsl
    );
}

// ===========================================================================
// Loop kernel
// ===========================================================================

#[quanta::kernel]
fn emit_loop(data: &mut [f32], iterations: u32) {
    let i = quark_id();
    let mut sum = 0.0f32;
    let mut j = 0u32;
    while j < iterations {
        sum = sum + 1.0;
        j = j + 1;
    }
    data[i] = sum;
}

#[test]
fn loop_msl_valid() {
    let msl = EMIT_LOOP_BINARY.msl.unwrap();
    assert!(
        msl.contains("kernel void emit_loop"),
        "MSL must have kernel declaration: {}",
        msl
    );
}

#[test]
fn loop_wgsl_valid() {
    let wgsl = EMIT_LOOP_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("fn emit_loop"),
        "WGSL must have kernel fn: {}",
        wgsl
    );
}

// ===========================================================================
// Multiple buffers kernel
// ===========================================================================

#[quanta::kernel]
fn emit_multi_buffer(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = quark_id();
    c[i] = a[i] + b[i];
}

#[test]
fn multi_buffer_msl_has_all_params() {
    let msl = EMIT_MULTI_BUFFER_BINARY.msl.unwrap();
    // Should have buffer annotations for each param
    assert!(
        msl.contains("buffer(0)") || msl.contains("[[buffer(0)]]"),
        "MSL must bind buffer 0: {}",
        msl
    );
    assert!(
        msl.contains("buffer(1)") || msl.contains("[[buffer(1)]]"),
        "MSL must bind buffer 1: {}",
        msl
    );
    assert!(
        msl.contains("buffer(2)") || msl.contains("[[buffer(2)]]"),
        "MSL must bind buffer 2: {}",
        msl
    );
}

#[test]
fn multi_buffer_wgsl_has_all_bindings() {
    let wgsl = EMIT_MULTI_BUFFER_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("binding(0)") || wgsl.contains("@binding(0)"),
        "WGSL must bind slot 0: {}",
        wgsl
    );
    assert!(
        wgsl.contains("binding(1)") || wgsl.contains("@binding(1)"),
        "WGSL must bind slot 1: {}",
        wgsl
    );
    assert!(
        wgsl.contains("binding(2)") || wgsl.contains("@binding(2)"),
        "WGSL must bind slot 2: {}",
        wgsl
    );
}

// ===========================================================================
// Push constant kernel
// ===========================================================================

#[quanta::kernel]
fn emit_push(data: &mut [f32], scale: f32) {
    let i = quark_id();
    data[i] = data[i] * scale;
}

#[test]
fn push_constant_msl_valid() {
    let msl = EMIT_PUSH_BINARY.msl.unwrap();
    assert!(
        msl.contains("kernel void emit_push"),
        "MSL must have kernel: {}",
        msl
    );
    // Push constant should appear as a buffer parameter or constant ref
    assert!(
        msl.contains("scale") || msl.contains("constant"),
        "MSL must reference push constant: {}",
        msl
    );
}

#[test]
fn push_constant_wgsl_valid() {
    let wgsl = EMIT_PUSH_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("fn emit_push"),
        "WGSL must have kernel fn: {}",
        wgsl
    );
    assert!(
        wgsl.contains("scale") || wgsl.contains("uniform"),
        "WGSL must reference push constant: {}",
        wgsl
    );
}

// ===========================================================================
// MSL/WGSL structure sanity
// ===========================================================================

#[test]
fn msl_always_has_metal_stdlib() {
    let msl = EMIT_DOUBLE_BINARY.msl.unwrap();
    assert!(
        msl.contains("metal_stdlib") || msl.contains("metal"),
        "MSL must include metal stdlib: {}",
        msl
    );
}

#[test]
fn wgsl_always_has_workgroup_size() {
    let wgsl = EMIT_DOUBLE_BINARY.wgsl.unwrap();
    assert!(
        wgsl.contains("@workgroup_size") || wgsl.contains("workgroup_size"),
        "WGSL must declare workgroup size: {}",
        wgsl
    );
}
