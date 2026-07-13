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

/// A backend selected by the `QUANTA_BACKEND` forcing lever.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForcedBackend {
    Metal,
    Vulkan,
    Cpu,
}

/// The accepted `QUANTA_BACKEND` values, in the order the error message
/// lists them. Kept as one source of truth for the parser and the
/// unknown-value diagnostic.
#[cfg(feature = "std")]
const QUANTA_BACKEND_VALUES: &str = "metal, vulkan, cpu";

/// Parse the `QUANTA_BACKEND` forcing lever.
///
/// `Ok(None)` — unset (normal per-OS discovery). `Ok(Some(_))` — a valid,
/// case-insensitive backend name. `Err(_)` — set to an unrecognized value;
/// the error names the accepted set. This never *checks availability*: a
/// backend that parses but is not present yields an empty device list, and
/// `init` turns that into the named "forced but unavailable" error.
#[cfg(feature = "std")]
fn forced_backend() -> Result<Option<ForcedBackend>, QuantaError> {
    match std::env::var("QUANTA_BACKEND") {
        Err(_) => Ok(None),
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "metal" => Ok(Some(ForcedBackend::Metal)),
            "vulkan" => Ok(Some(ForcedBackend::Vulkan)),
            "cpu" => Ok(Some(ForcedBackend::Cpu)),
            other => Err(QuantaError::invalid_param(alloc::format!(
                "QUANTA_BACKEND={other:?} is not a recognized backend \
                 (accepted values: {QUANTA_BACKEND_VALUES})"
            ))),
        },
    }
}

/// Discover available GPU devices, in the platform's fixed preference
/// order (see [`init`] for the contract).
///
/// **This is the programmatic device-selection path.** There is no
/// separate "pick a device" API: enumerate here and choose from the
/// returned list (by index, name, or capability query) — [`init`] is
/// merely the "take the first" convenience over this.
///
/// # The `QUANTA_BACKEND` forcing lever
///
/// When `QUANTA_BACKEND` is set to `metal`, `vulkan`, or `cpu`
/// (case-insensitive), this returns **only** that backend's devices —
/// the others are not even probed. A backend that is forced but not
/// present (e.g. `QUANTA_BACKEND=vulkan` where no Vulkan loader exists)
/// yields an **empty** list rather than silently falling through to
/// another backend; CI determinism is the point. `QUANTA_BACKEND=cpu`
/// includes the CPU software device regardless of `QUANTA_CPU`. An
/// unrecognized value yields an empty list here (and a named error from
/// [`init`]).
#[cfg(feature = "std")]
pub fn devices() -> alloc::vec::Vec<Gpu> {
    // An unrecognized value parses to Err; treat it as "select nothing"
    // here so no backend leaks through. `init` reports the named error.
    let forced = forced_backend().ok().flatten();
    // Unused when no discovery-bearing feature is enabled (e.g. a bare
    // `std` or wasm32+webgpu build compiles none of the cfg-gated probes
    // below); the closure captures `forced`, keeping it "used" too.
    #[allow(unused_variables)]
    let allows = |b: ForcedBackend| forced.is_none() || forced == Some(b);

    // `mut` is conditional: only the metal/vulkan/software cfgs below mutate
    // the vector, and feature combinations may disable all of them (e.g.
    // wasm32 + webgpu).
    #[allow(unused_mut)]
    let mut devs: alloc::vec::Vec<alloc::boxed::Box<dyn GpuDevice>> = alloc::vec::Vec::new();

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    if allows(ForcedBackend::Metal) {
        devs.extend(driver::metal::discover());
    }

    #[cfg(all(
        feature = "vulkan",
        any(
            target_os = "linux",
            target_os = "android",
            target_os = "windows",
            all(feature = "vulkan-portability", target_os = "macos"),
        )
    ))]
    if allows(ForcedBackend::Vulkan) {
        devs.extend(driver::vulkan::discover());
    }

    // The CPU device joins normal discovery only under QUANTA_CPU=1, but
    // forcing QUANTA_BACKEND=cpu selects it unconditionally.
    #[cfg(feature = "software")]
    {
        let cpu_opt_in = std::env::var("QUANTA_CPU")
            .map(|v| v == "1")
            .unwrap_or(false);
        if forced == Some(ForcedBackend::Cpu) || (forced.is_none() && cpu_opt_in) {
            devs.extend(driver::cpu::discover());
        }
    }

    devs.into_iter().map(maybe_validate).map(Gpu::new).collect()
}

/// Initialize the first available GPU device, following Quanta's fixed
/// per-OS discovery contract.
///
/// # Discovery order (the contract)
///
/// Probing order is deliberate and stable — not an accident of
/// enumeration — so a given machine always initializes the same backend:
///
/// - **Apple (macOS / iOS):** Metal, then the CPU software device when
///   `QUANTA_CPU=1`.
/// - **Linux / Android / Windows:** Vulkan, then the CPU software device
///   when `QUANTA_CPU=1`.
/// - **wasm:** WebGPU — through the async [`init_webgpu_async`]; sync
///   `init` never returns a WebGPU device (the platform requires an async
///   adapter handshake).
///
/// `init` returns the first device [`devices`] enumerates in that order.
/// To choose a different one, call [`devices`] and pick from the list —
/// that is the programmatic override; there is no separate selection API.
///
/// # The `QUANTA_BACKEND` forcing lever
///
/// Set `QUANTA_BACKEND` to `metal`, `vulkan`, or `cpu` (case-insensitive)
/// to restrict discovery to exactly that backend. A backend that is forced
/// but unavailable does **not** fall through to another: `init` fails with
/// an error naming the env var, so CI never silently runs on the wrong
/// backend. An unrecognized value fails with an error listing the accepted
/// values.
#[cfg(feature = "std")]
pub fn init() -> Result<Gpu, QuantaError> {
    // Surface an unrecognized QUANTA_BACKEND as its own named error before
    // anything else — the value is a mistake regardless of what hardware
    // is present.
    let forced = forced_backend()?;

    let mut devs = devices();
    if let Some(dev) = devs.drain(..).next() {
        Ok(dev)
    } else if let Some(backend) = forced {
        Err(QuantaError::not_supported(alloc::format!(
            "QUANTA_BACKEND forces the {} backend, but no {} device is \
             available on this machine (discovery will not fall through to \
             another backend)",
            forced_backend_name(backend),
            forced_backend_name(backend),
        )))
    } else {
        Err(QuantaError::no_device())
    }
}

/// The lowercase name of a forced backend, for diagnostics.
#[cfg(feature = "std")]
fn forced_backend_name(backend: ForcedBackend) -> &'static str {
    match backend {
        ForcedBackend::Metal => "metal",
        ForcedBackend::Vulkan => "vulkan",
        ForcedBackend::Cpu => "cpu",
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
