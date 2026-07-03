//! KernelDef → Vulkan SPIR-V binary.
//!
//! Walks KernelOps and produces valid Vulkan SPIR-V binary (Shader capability,
//! GLCompute execution model, StorageBuffer storage class). This replaces the
//! LLVM spirv64 backend which emits OpenCL-style SPIR-V that Vulkan rejects.
//!
//! The output is a `Vec<u8>` ready for `vkCreateShaderModule`.

mod constants;
mod emitter;
mod kernel;
mod ops;
mod shader;
mod types;

use emitter::SpvEmitter;

/// Emit Vulkan SPIR-V binary from a KernelDef.
///
/// Narrow scalar buffers use native stride (bf16 → 16-bit, fp8 → 8-bit
/// elements) — the storage contract shared with the host upload and the CPU
/// executor. The module declares `StorageBuffer16BitAccess` /
/// `StorageBuffer8BitAccess` only when the kernel touches those dtypes;
/// devices lacking the matching Vulkan features reject the pipeline (the
/// caps table marks bf16/fp8 as feature-gated on Vulkan).
///
/// Returns the SPIR-V module as bytes, ready for `vkCreateShaderModule`.
pub fn emit(kernel: &crate::KernelDef) -> Result<Vec<u8>, String> {
    let mut e = SpvEmitter::new();
    e.emit_kernel(kernel)?;
    Ok(e.finalize())
}

/// Emit SPIR-V for a vertex shader from a [`ShaderDef`].
pub fn emit_vertex(shader: &crate::ShaderDef) -> Result<Vec<u8>, String> {
    let mut e = SpvEmitter::new();
    e.emit_vertex_shader(shader)?;
    Ok(e.finalize())
}

/// Emit SPIR-V for a fragment shader from a [`ShaderDef`].
pub fn emit_fragment(shader: &crate::ShaderDef) -> Result<Vec<u8>, String> {
    let mut e = SpvEmitter::new();
    e.emit_fragment_shader(shader)?;
    Ok(e.finalize())
}
