//! CPU software driver — executes KernelDef IR without GPU hardware.
//!
//! Simulates GPU execution on CPU by walking KernelOp instructions
//! sequentially per thread. Enables:
//! - Testing without GPU hardware
//! - CI on any machine
//! - Debugging kernels step by step
//! - CPU oracle for correctness verification

mod device;
mod eval;
mod exec;
mod value;

pub use device::{CpuDevice, discover};
