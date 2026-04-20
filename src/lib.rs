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

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

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

/// Returns true if the `QUANTA_VALIDATE` env var is set to "1".
#[cfg(feature = "std")]
fn validation_enabled() -> bool {
    std::env::var("QUANTA_VALIDATE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Optionally wrap a device in the validation layer.
#[cfg(feature = "std")]
fn maybe_validate(dev: alloc::boxed::Box<dyn GpuDevice>) -> alloc::boxed::Box<dyn GpuDevice> {
    if validation_enabled() {
        driver::validation::ValidationDevice::wrap(dev)
    } else {
        dev
    }
}

/// Discover available GPU devices.
#[cfg(feature = "std")]
pub fn devices() -> alloc::vec::Vec<Gpu> {
    let mut devs: alloc::vec::Vec<alloc::boxed::Box<dyn GpuDevice>> = alloc::vec::Vec::new();

    #[cfg(feature = "metal")]
    devs.extend(driver::metal::discover());

    #[cfg(feature = "vulkan")]
    devs.extend(driver::vulkan::discover());

    #[cfg(feature = "software")]
    devs.extend(driver::software::discover());

    devs.into_iter().map(maybe_validate).map(Gpu::new).collect()
}

/// Initialize the first available GPU device.
#[cfg(feature = "std")]
pub fn init() -> Result<Gpu, QuantaError> {
    let mut devs = devices();
    if devs.is_empty() {
        Err(QuantaError::no_device())
    } else {
        Ok(devs.remove(0))
    }
}
