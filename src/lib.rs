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
//!
//! ## Crate layout
//!
//! This crate is the facade over the split substrate:
//!
//! - `quanta-core` — the shared device line (sealed `GpuDevice`,
//!   drivers, fields/textures/sync). Re-exported wholesale here.
//! - the compute face — `#[quanta::kernel]` / waves / the scan
//!   library, compiled in behind the `compute` feature.
//! - `quanta-render` — the render face (`RenderGpu`, typed wrappers,
//!   render-stage macros), re-exported here behind the `render`
//!   feature. Render-only consumers (e.g. a UI toolkit) can depend on
//!   `quanta-render` directly and never pull the compute stack into
//!   their dependency graph.

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "compute")]
pub mod kernel;
#[cfg(feature = "compute")]
pub mod scan;

/// GPU intrinsics — `extern "C"` imports surfaced as
/// `import "quanta" "<name>"` in the WASM emitted by
/// `#[quanta::kernel]`. cfg-gated to wasm32; the lowering pass on the
/// host side resolves them. See roadmap step 058.
#[cfg(feature = "compute")]
pub mod intrinsics;

/// Host-side stubs for every GPU intrinsic, used by `_src!()` macros
/// emitted by `#[quanta::device]`. Hidden from the public API — the
/// `_src!` macro injects `use ::quanta::__device_host_stubs::*` inside
/// its `const _: () = { ... }` block so spliced device-fn bodies
/// name-resolve in any downstream crate without the user importing
/// anything.
#[cfg(feature = "compute")]
#[doc(hidden)]
pub mod __device_host_stubs;

/// Spec enum tables for the WebGPU IDL (B′ track of FFI TCB shrink).
/// Generated from `web/webgpu.idl` by `quanta codegen webgpu`; the
/// `tests` block inside checks that every enum string Quanta hands
/// the JS side is a member of the spec table. Top-level so native
/// `cargo test` runs the subset check without needing a wasm32 build.
pub mod webgpu_generated_codes;

// The shared substrate: device discovery (`init` / `devices` /
// `init_cpu`), the `Gpu` handle, fields, textures, sync, and — under
// the matching features — the compute / render data models.
pub use quanta_core::*;

// Re-export kernel types. (`ShaderBinary` / `ShaderStage` are *render*
// types — they live on the render surface, re-exported at the root
// only when the `render` feature is on.)
//
// `ScalarType` is a `quanta-ir` kernel-language type (the scalar tag
// carried by `GpuType::scalar_type()`); it is re-exported at the root
// under the `compute` feature because `#[quanta::kernel]`-generated
// code names it through `quanta::ScalarType`.
#[cfg(feature = "compute")]
pub use kernel::{GpuType, KernelBinary, ScalarType};

// Compute-face proc macros — only when the `compute` feature is on
// (symmetric to the render-stage macros below).
#[cfg(feature = "compute")]
pub use quanta_dsl::__kernel_inner;
#[cfg(feature = "compute")]
pub use quanta_dsl::device;
#[cfg(feature = "compute")]
pub use quanta_dsl::gpu_type;
#[cfg(feature = "compute")]
pub use quanta_dsl::import_devices;
#[cfg(feature = "compute")]
pub use quanta_dsl::kernel;

// Derive macros (compute face)
#[cfg(feature = "compute")]
pub use quanta_dsl::Fields;
#[cfg(feature = "compute")]
pub use quanta_dsl::Uniforms;

// Render face — the `quanta-render` crate, re-exported wholesale so
// `use quanta::*` covers render consumers that come through the
// facade. This brings in the render-stage shader macros, the typed
// render wrappers, and the `RenderGpu` extension trait alongside the
// render data model re-exported from `quanta-core` above.
#[cfg(feature = "render")]
pub use quanta_render::*;
