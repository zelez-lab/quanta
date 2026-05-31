//! Quanta kernel intermediate representation.
//!
//! Shared between the proc macro (`quanta-macros`) and the compiler binary
//! (`quanta-compiler`). Defines the platform-agnostic IR that represents
//! GPU kernels between parsing and code generation.

pub mod caps;
pub mod output;
pub mod scope_check;
pub mod serial;
pub mod shader;
pub mod types;
pub mod validate;
pub mod wire;

/// SPIR-V emitter for JIT compilation. Enabled by the `jit` feature.
#[cfg(feature = "jit")]
pub mod emit_spirv;

/// MSL emitter for JIT compilation. Enabled by the `jit` feature.
#[cfg(feature = "jit")]
pub mod emit_msl;

/// WGSL emitter for JIT compilation. Enabled by the `jit` feature.
#[cfg(feature = "jit")]
pub mod emit_wgsl;

// ── Re-exports for backward compatibility ────────────────────────────────────

pub use output::CompilerOutput;
pub use serial::{
    deserialize_kernel, deserialize_output, deserialize_shader, deserialize_shader_output,
    serialize_kernel, serialize_output, serialize_shader, serialize_shader_output,
};
pub use shader::{ShaderDef, ShaderOutput, ShaderParam, ShaderStage, ShaderType};
pub use types::{
    AtomicOp, BinOp, CmpOp, ConstValue, DeviceFnDef, KernelDef, KernelOp, KernelParam, MathFn,
    MemoryOrder, Reg, ScalarType, UnaryOp,
};
