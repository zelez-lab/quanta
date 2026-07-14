//! # Quanta kernel language
//!
//! GPU kernels are written as annotated Rust functions. The `#[quanta::kernel]`
//! proc macro compiles them to GPU code at build time.
//!
//! See `quanta-ir` crate for the full KernelIR specification.
//! See module-level docs for the kernel language reference.

// Re-export IR types from the shared crate
pub use quanta_ir::{
    AtomicOp, BinOp, CmpOp, CompilerOutput, ConstValue, KernelDef, KernelOp, KernelParam, MathFn,
    MatrixFrag, MemoryOrder, Reg, ScalarType, UnaryOp,
};

// `GpuType` (the kernel-type marker trait) and `KernelBinary` (the
// compiled-kernel struct a `#[quanta::kernel]` expands to) moved down to
// `quanta-core` (behind `compute`) so companion crates that host kernels
// reach them through `quanta-core` and no longer depend on this facade.
// Re-exported here so the facade's public surface is unchanged — end
// users still name `quanta::GpuType` / `quanta::KernelBinary`.
pub use quanta_core::{GpuType, KernelBinary};
