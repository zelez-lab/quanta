//! Quanta kernel intermediate representation.
//!
//! Shared between the proc macro (`quanta-macros`) and the compiler binary
//! (`quanta-compiler`). Defines the platform-agnostic IR that represents
//! GPU kernels between parsing and code generation.

pub mod caps;
pub mod const_analysis;
pub mod dtype;
pub mod dtype_codegen;
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

/// Device capabilities that steer emitter code paths (e.g. native bf16
/// storage vs. the portable u32-word fallback). Enabled by `jit`.
#[cfg(feature = "jit")]
pub mod emit_caps;

/// Per-op differential matrix case generator. Shared source of truth for
/// the `op_matrix` test harness and the WGSL browser audit. Enabled by
/// the `op-matrix-cases` feature.
#[cfg(feature = "op-matrix-cases")]
pub mod op_matrix_cases;

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
