//! # Quanta core — the shared GPU substrate
//!
//! The device line every Quanta face is built on: the sealed
//! [`GpuDevice`] trait, the in-tree drivers (Metal / Vulkan / CPU
//! software / WebGPU), and the resource surface they speak — fields,
//! textures, samplers, pulses, timelines, queries.
//!
//! Consumers do not depend on this crate directly:
//!
//! - the `quanta` facade re-exports the whole surface and adds the
//!   compute face (`#[quanta::kernel]`, waves, the scan library);
//! - `quanta-render` builds on the `render` feature and adds the
//!   render face (`RenderGpu`, builders, typed render wrappers).
//!
//! The `render` / `compute` Cargo features gate the two halves of the
//! device trait and of each driver. The render *data model* the trait
//! and drivers speak (`PipelineDesc`, `RenderPass`, `RenderOp`,
//! shader binaries, surface configuration) lives here behind
//! `render`; the compute data model (`Wave`, `Batch`, queues,
//! compute ICBs) lives here behind `compute`.

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod api;
mod driver;

// Re-export API types at crate root
pub use api::*;

/// Returns true if the `QUANTA_VALIDATE` env var is set to "1".
#[cfg(feature = "std")]
fn validation_enabled() -> bool {
    std::env::var("QUANTA_VALIDATE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Optionally wrap a device in the validation layer.
#[cfg(feature = "std")]
fn maybe_validate(dev: alloc::boxed::Box<dyn GpuDevice>) -> alloc::sync::Arc<dyn GpuDevice> {
    if validation_enabled() {
        alloc::sync::Arc::from(driver::validation::ValidationDevice::wrap(dev))
    } else {
        alloc::sync::Arc::from(dev)
    }
}

/// Discover available GPU devices.
#[cfg(feature = "std")]
pub fn devices() -> alloc::vec::Vec<Gpu> {
    // `mut` is conditional: only the metal/vulkan/software cfgs below mutate
    // the vector, and feature combinations may disable all of them (e.g.
    // wasm32 + webgpu).
    #[allow(unused_mut)]
    let mut devs: alloc::vec::Vec<alloc::boxed::Box<dyn GpuDevice>> = alloc::vec::Vec::new();

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    devs.extend(driver::metal::discover());

    #[cfg(feature = "vulkan")]
    devs.extend(driver::vulkan::discover());

    // Include CPU device if QUANTA_CPU=1 env var is set
    #[cfg(feature = "software")]
    {
        if std::env::var("QUANTA_CPU")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            devs.extend(driver::cpu::discover());
        }
    }

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

/// Initialize a CPU software device for testing without GPU hardware.
///
/// The CPU device executes kernel IR (KernelDef) sequentially on CPU.
/// Only supports the JIT path (`wave_jit`). Pre-compiled binaries
/// (SPIR-V, metallib) are not supported.
#[cfg(feature = "software")]
pub fn init_cpu() -> Gpu {
    let dev: alloc::boxed::Box<dyn GpuDevice> =
        alloc::boxed::Box::new(driver::cpu::CpuDevice::new());
    Gpu::new(maybe_validate(dev))
}

/// Initialize a WebGPU device. Browser-only. Async because the WebGPU
/// device is acquired through Promises (`navigator.gpu.requestAdapter`,
/// `adapter.requestDevice`).
///
/// This is the only entry point for the WebGPU driver — sync `init()`
/// can never return a WebGPU device because the platform requires an
/// async handshake.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub async fn init_webgpu_async() -> Result<Gpu, QuantaError> {
    driver::webgpu::init_async().await
}

/// Re-export of the WebGPU driver module for callers that need direct
/// access to `WebgpuDevice` (and its async extensions like
/// `field_read_bytes_async`).
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod webgpu {
    pub use crate::driver::webgpu::*;
}
