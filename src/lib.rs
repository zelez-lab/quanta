//! # Quanta — GPU compute and rendering API
//!
//! Quarks, protons, nuclei, fields, waves, pulses.
//!
//! One API (~20 functions), any GPU. Drivers are thin translation layers
//! to platform-specific backends (Metal on macOS/iOS, Vulkan on Linux/Android/Windows,
//! WebGPU in browsers).
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
pub mod scan;

/// GPU intrinsics — `extern "C"` imports surfaced as
/// `import "quanta" "<name>"` in the WASM emitted by
/// `#[quanta::kernel]`. cfg-gated to wasm32; the lowering pass on the
/// host side resolves them. See roadmap step 058.
pub mod intrinsics;

/// Host-side stubs for every GPU intrinsic, used by `_src!()` macros
/// emitted by `#[quanta::device]`. Hidden from the public API — the
/// `_src!` macro injects `use ::quanta::__device_host_stubs::*` inside
/// its `const _: () = { ... }` block so spliced device-fn bodies
/// name-resolve in any downstream crate without the user importing
/// anything.
#[doc(hidden)]
pub mod __device_host_stubs;

/// Spec enum tables for the WebGPU IDL (B′ track of FFI TCB shrink).
/// Generated from `web/webgpu.idl` by `quanta codegen webgpu`; the
/// `tests` block inside checks that every enum string Quanta hands
/// the JS side is a member of the spec table. Top-level so native
/// `cargo test` runs the subset check without needing a wasm32 build.
pub mod webgpu_generated_codes;

// Re-export API types at crate root
pub use api::*;

// Re-export kernel types
pub use kernel::{GpuType, KernelBinary, ScalarType, ShaderBinary, ShaderStage};

// Re-export proc macros (compute — always available)
pub use quanta_macros::__kernel_inner;
pub use quanta_macros::device;
pub use quanta_macros::gpu_type;
pub use quanta_macros::import_devices;
pub use quanta_macros::kernel;

// Derive macros (compute / shared)
pub use quanta_macros::Fields;
pub use quanta_macros::Uniforms;

// Render-stage shader macros — only when the `render` feature is on.
// (`quanta-render` enables it; a headless build omits them entirely.)
#[cfg(feature = "render")]
pub use quanta_macros::Vertex;
#[cfg(feature = "render")]
pub use quanta_macros::closest_hit;
#[cfg(feature = "render")]
pub use quanta_macros::fragment;
#[cfg(feature = "render")]
pub use quanta_macros::mesh;
#[cfg(feature = "render")]
pub use quanta_macros::miss;
#[cfg(feature = "render")]
pub use quanta_macros::ray_gen;
#[cfg(feature = "render")]
pub use quanta_macros::task;
#[cfg(feature = "render")]
pub use quanta_macros::tess_control;
#[cfg(feature = "render")]
pub use quanta_macros::tess_eval;
#[cfg(feature = "render")]
pub use quanta_macros::vertex;

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

    #[cfg(all(feature = "metal", target_os = "macos"))]
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
