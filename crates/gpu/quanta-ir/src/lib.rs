//! Quanta kernel intermediate representation.
//!
//! Shared between the DSL proc macros (`quanta-compute-dsl` /
//! `quanta-render-dsl`, through `quanta-dsl-core`) and the compiler binary
//! (`quanta-compiler`). Defines the platform-agnostic IR that represents
//! GPU kernels between parsing and code generation.

pub mod caps;
pub mod const_analysis;
pub mod dispatch_fold;
pub mod dtype;
pub mod dtype_codegen;
pub mod output;
pub mod quant;
pub mod reg_mutability;
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

/// Per-op differential matrix case generator. Shared source of truth for
/// the `op_matrix` test harness and the WGSL browser audit. Enabled by
/// the `op-matrix-cases` feature.
#[cfg(feature = "op-matrix-cases")]
pub mod op_matrix_cases;

// ── Re-exports for backward compatibility ────────────────────────────────────

pub use output::CompilerOutput;
pub use quant::{QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue};
pub use reg_mutability::collect_mutable_regs;
pub use serial::{
    deserialize_kernel, deserialize_output, deserialize_shader, deserialize_shader_output,
    serialize_kernel, serialize_output, serialize_shader, serialize_shader_output,
};
pub use shader::{
    ShaderDef, ShaderOutput, ShaderParam, ShaderStage, ShaderType, ShaderVaryings, VaryingField,
};
pub use types::{
    AtomicOp, BinOp, CmpOp, ConstValue, DeviceFnDef, KernelDef, KernelOp, KernelParam, MathFn,
    MatrixFrag, MemoryOrder, Reg, ScalarType, UnaryOp, is_f64_transcendental,
};
