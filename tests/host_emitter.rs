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
//
// `while`-loop kernels previously hit bug #1 (r44 used before
// definition at Branch.cond via the WASM-route's
// `install_redirect_at` path). Fixed by the backward-slice hoist in
// `crates/quanta-wasm-lowering/src/lower.rs::hoist_cond_defining_ops`
// that moves the cond-defining ops from the current frame's sink to
// the target frame's sink before installing the redirect Branch.

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

// ===========================================================================
// Explicit memory fence (D-ext.3a + D-ext.3b.3)
// ===========================================================================
//
// Exercises the proc-macro `fence(MemoryOrder::Release)` builtin: the macro
// must accept a fence call with any of the five MemoryOrder variants and
// emit a corresponding KernelOp::Fence in the kernel body. We sandwich a
// fence between two atomic ops to mirror the future release/acquire
// litmus pattern.

#[quanta::kernel]
fn emit_fence(flag: &mut [u32], data: &mut [u32]) {
    let i = quark_id();
    data[i] = atomic_add(&mut flag[0], 1);
    fence(Release);
    data[i] = atomic_add(&mut flag[0], 1);
}

#[test]
fn fence_kernel_compiles() {
    let _binary = &EMIT_FENCE_BINARY;
}
