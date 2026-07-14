//! KernelDef → WebGPU Shading Language (JIT path).
//!
//! Walks KernelOps and produces valid WGSL. Parallel to [`emit_msl`] and
//! [`emit_spirv`], optimized for size — this emitter ships inside the wasm
//! binary served to browsers, where every byte counts.
//!
//! Coverage: every [`KernelOp`] variant produces equivalent WGSL.
//! Cross-checked against the build-time emitter in `quanta-compiler` and
//! against the Kani exhaustiveness theorems T1001.

mod helpers;
mod kernel;
mod ops;
mod shader;
mod shader_tokenizer;
mod shader_walker;

pub use kernel::emit;
pub use shader::{emit_fragment_shader, emit_vertex_shader};

/// Emit WGSL source for a kernel at runtime.
///
/// Wrapper that matches the naming used in [step 050]'s WebGPU driver and
/// the 079 README. Returns the WGSL string ready for
/// `GPUDevice.createShaderModule({ code })`.
pub fn emit_wgsl_jit(kernel: &crate::KernelDef) -> Result<String, String> {
    emit(kernel)
}
