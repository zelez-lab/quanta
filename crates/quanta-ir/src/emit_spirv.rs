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
