//! # Quanta — GPU compute and rendering API
//!
//! Quarks, protons, nuclei, fields, waves, pulses.
//!
//! One API (~20 functions), any GPU. Drivers are thin translation layers
//! to platform-specific backends (compiled drivers on Zelez, Metal on macOS,
//! Vulkan on Linux/Windows, WebGPU in browsers).
//!
//! ## Naming (subatomic physics)
//!
//! | Name     | GPU hardware       | Description                          |
//! |----------|--------------------|--------------------------------------|
//! | Quark    | Thread / lane      | Smallest unit of execution           |
//! | Proton   | Core               | Executes quarks                      |
//! | Nucleus  | Compute unit       | Group of protons                     |
//! | Field    | Buffer             | Data that quarks operate on          |
//! | Wave     | Compute dispatch   | A bound kernel ready to dispatch     |
//! | Pulse    | Fence / sync       | GPU completion signal                |
//! | Quanta   | The whole GPU      | All nuclei firing together           |
//!
//! ## Kernel language
//!
//! GPU kernels are written as annotated Rust functions:
//!
//! ```ignore
//! #[quanta::kernel]
//! fn scan_filter(input: &[f32], output: &mut [f32], threshold: f32) {
//!     let i = quark_id();
//!     if input[i] > threshold {
//!         output[i] = input[i];
//!     }
//! }
//! ```
//!
//! The proc macro compiles directly to GPU ISA at build time:
//! - AMD: LLVM → amdgcn binary
//! - NVIDIA: LLVM → nvptx64 PTX
//! - Apple: generates MSL source string
//! - Browser: generates WGSL source string
//!
//! No SPIR-V. No intermediate representation. Direct to target.

mod api;
mod driver;
pub mod kernel;

// Re-export API types at crate root
pub use api::*;

// Re-export kernel types
pub use kernel::{GpuType, KernelBinary};

// Re-export proc macros
pub use quanta_macros::device;
pub use quanta_macros::kernel;

/// Discover available GPU devices.
pub fn devices() -> Vec<Gpu> {
    let mut devs: Vec<Box<dyn GpuDevice>> = Vec::new();

    #[cfg(feature = "metal")]
    devs.extend(driver::metal::discover());

    #[cfg(feature = "software")]
    devs.extend(driver::software::discover());

    devs.into_iter().map(Gpu::new).collect()
}

/// Initialize the first available GPU device.
pub fn init() -> Result<Gpu, QuantaError> {
    let mut devs = devices();
    if devs.is_empty() {
        Err(QuantaError::NoDevice)
    } else {
        Ok(devs.remove(0))
    }
}
