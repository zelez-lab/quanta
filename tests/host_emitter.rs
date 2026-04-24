//! Tier 1 (host, no GPU) conformance tests — kernel proc macro compilation.
//!
//! Verifies that various kernel patterns compile successfully through the
//! proc macro. MSL/WGSL text output was removed in the binary-only refactor;
//! these tests now verify that kernel binaries are produced without errors.
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
fn basic_kernel_compiles() {
    // The kernel macro produces a KernelBinary static.
    // Without quanta-compiler, all binary fields are None.
    let _binary = &EMIT_DOUBLE_BINARY;
}

// ===========================================================================
// Shared memory kernel
// ===========================================================================

#[quanta::kernel]
fn emit_shared(input: &[f32], output: &mut [f32]) {
    #[quanta::shared]
    let local: [f32; 256];
    let lid = proton_id();
    let gid = quark_id();
    local[lid] = input[gid];
    barrier();
    output[gid] = local[lid];
}

#[test]
fn shared_kernel_compiles() {
    let _binary = &EMIT_SHARED_BINARY;
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
fn atomic_kernel_compiles() {
    let _binary = &EMIT_ATOMIC_BINARY;
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
fn branch_kernel_compiles() {
    let _binary = &EMIT_BRANCH_BINARY;
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
fn device_fn_kernel_compiles() {
    let _binary = &EMIT_WITH_DEVICE_BINARY;
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
fn loop_kernel_compiles() {
    let _binary = &EMIT_LOOP_BINARY;
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
fn multi_buffer_kernel_compiles() {
    let _binary = &EMIT_MULTI_BUFFER_BINARY;
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
fn push_constant_kernel_compiles() {
    let _binary = &EMIT_PUSH_BINARY;
}
