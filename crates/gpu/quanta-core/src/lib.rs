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

/// Re-export of `raw-window-handle` 0.6 (feature `raw-window-handle`),
/// so consumers of `SurfaceTarget::from_window` / `from_raw` name the
/// interop types (`rwh::HasWindowHandle`, `rwh::RawWindowHandle`, …)
/// without a dependency line of their own.
#[cfg(feature = "raw-window-handle")]
pub use raw_window_handle as rwh;

/// Returns true if the `QUANTA_VALIDATE` env var is set to "1".
// Reached only through `maybe_validate` — see its note.
#[cfg(feature = "std")]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn validation_enabled() -> bool {
    std::env::var("QUANTA_VALIDATE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Optionally wrap a device in the validation layer, then hand the
/// device a weak reference to its own `Arc` so the pulses it mints can
/// keep it alive (see `GpuDevice::install_self_ref`).
// Every call site is a native one — `init_isolated`'s metal/vulkan/
// software arms, `init_cpu_isolated`, and the registry's device build;
// wasm32 reaches its device through the WebGPU async handshake instead.
#[cfg(feature = "std")]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn maybe_validate(dev: alloc::boxed::Box<dyn GpuDevice>) -> alloc::sync::Arc<dyn GpuDevice> {
    let arc: alloc::sync::Arc<dyn GpuDevice> = if validation_enabled() {
        alloc::sync::Arc::from(driver::validation::ValidationDevice::wrap(dev))
    } else {
        alloc::sync::Arc::from(dev)
    };
    arc.install_self_ref(alloc::sync::Arc::downgrade(&arc));
    arc
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
///
/// # The software last resort
///
/// When nothing is forced and no GPU backend yields a device, the CPU
/// software device joins as a last resort (requires the `software`
/// feature) — announced by a loud `quanta:` line on stderr, following
/// the same discipline as the per-backend unavailability lines (e.g.
/// the Vulkan driver naming a missing `vulkan-1.dll`). A machine
/// without GPU drivers still gets a working device, but never
/// silently. `QUANTA_CPU=1` is different: it opts the CPU device into
/// normal discovery *alongside* GPUs.
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

    // Every backend goes through the process-wide device registry
    // (`api::registry`): repeated `devices()`/`init()` calls converge
    // on the SAME live devices (Gpu clones of one shared context per
    // physical device) instead of re-running discovery — see the
    // registry module for the lifetime story. `mut` is conditional:
    // only the metal/vulkan/software cfgs below mutate the vector, and
    // feature combinations may disable all of them (e.g. wasm32 +
    // webgpu).
    // The import goes unused for that same reason.
    #[cfg_attr(target_arch = "wasm32", allow(unused_imports))]
    use api::registry::{BackendKind, get_or_discover};
    #[allow(unused_mut)]
    let mut devs: alloc::vec::Vec<Gpu> = alloc::vec::Vec::new();

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    if allows(ForcedBackend::Metal) {
        devs.extend(get_or_discover(BackendKind::Metal, driver::metal::discover));
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
        devs.extend(get_or_discover(
            BackendKind::Vulkan,
            driver::vulkan::discover,
        ));
    }

    // The CPU device joins normal discovery only under QUANTA_CPU=1, but
    // forcing QUANTA_BACKEND=cpu selects it unconditionally.
    #[cfg(feature = "software")]
    {
        let cpu_opt_in = std::env::var("QUANTA_CPU")
            .map(|v| v == "1")
            .unwrap_or(false);
        if forced == Some(ForcedBackend::Cpu) || (forced.is_none() && cpu_opt_in) {
            devs.extend(get_or_discover(BackendKind::Cpu, driver::cpu::discover));
        }
    }

    // Last-resort software fallback: with nothing forced and no GPU
    // backend found, the CPU device joins so a machine without GPU
    // drivers still initializes — loudly, because a silently-engaged
    // software lane looks exactly like a healthy GPU lane. Forcing is
    // exempt on purpose: QUANTA_BACKEND pins one backend and must fail
    // rather than fall through (CI determinism).
    #[cfg(feature = "software")]
    if devs.is_empty() && forced.is_none() {
        std::eprintln!(
            "quanta: no GPU backend available — falling back to SOFTWARE \
             rendering (CPU device)"
        );
        devs.extend(get_or_discover(BackendKind::Cpu, driver::cpu::discover));
    }

    devs
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
/// - **Last resort (all native platforms):** when nothing is forced and
///   no GPU backend is present, the CPU software device engages with a
///   loud stderr line (see [`devices`]) — so `init` succeeds on a
///   machine without GPU drivers instead of failing before the app can
///   draw anything.
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
    match devs.drain(..).next() {
        Some(dev) => Ok(dev),
        None => Err(no_device_after_discovery(forced)),
    }
}

/// The `init` family's terminal error when discovery yields nothing.
/// With the `QUANTA_BACKEND` lever set, names the forced backend's
/// actual gap: a forced backend can be missing from this machine (no
/// driver) or from this BUILD (feature not compiled in / platform
/// never compiles it). Blaming the machine for a feature gap sends
/// users hunting for drivers they don't need — e.g.
/// `QUANTA_BACKEND=cpu` in a build without `software`.
#[cfg(feature = "std")]
fn no_device_after_discovery(forced: Option<ForcedBackend>) -> QuantaError {
    let Some(backend) = forced else {
        return QuantaError::no_device();
    };
    let name = forced_backend_name(backend);
    QuantaError::not_supported(match forced_backend_build_gap(backend) {
        Some(gap) => alloc::format!(
            "QUANTA_BACKEND forces the {name} backend, but {gap} \
             (discovery will not fall through to another backend)"
        ),
        None => alloc::format!(
            "QUANTA_BACKEND forces the {name} backend, but no {name} \
             device is available on this machine (discovery will not \
             fall through to another backend)"
        ),
    })
}

/// Test-support constructor: initialize a PRIVATE device, bypassing
/// the process-wide device registry.
///
/// Same discovery contract as [`init`] — per-OS probe order, the
/// `QUANTA_BACKEND` forcing lever, `QUANTA_VALIDATE` wrapping, the
/// loud software last resort — but the registry is never read or
/// written: the returned `Gpu` (and its clones) hold a freshly built
/// device with its own deferred lane and MSAA pool, which no other
/// initialization path in the process can observe or share. The
/// shared registry device, if live, is untouched.
///
/// This exists for the lifecycle/leak suites: absolute
/// `debug_registry_counts` snapshots are only sound on a device no
/// parallel test can allocate on, and the shared `init()` device
/// stopped being that when the registry made every `init()` converge
/// on one device per process. Every call builds a real driver device,
/// so per-test use recreates the pre-registry driver load — reserve
/// it for count-asserting tests. Production code wants the registry
/// semantics: use [`init`].
#[cfg(feature = "std")]
#[doc(hidden)]
pub fn init_isolated() -> Result<Gpu, QuantaError> {
    let forced = forced_backend()?;
    // Unused when no discovery-bearing feature is enabled, same as in
    // `devices()`.
    #[allow(unused_variables)]
    let allows = |b: ForcedBackend| forced.is_none() || forced == Some(b);

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    if allows(ForcedBackend::Metal)
        && let Some(dev) = driver::metal::discover().into_iter().next()
    {
        return Ok(Gpu::new(maybe_validate(dev)));
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
    if allows(ForcedBackend::Vulkan)
        && let Some(dev) = driver::vulkan::discover().into_iter().next()
    {
        return Ok(Gpu::new(maybe_validate(dev)));
    }

    // The CPU software device: when forced, or as the last resort —
    // an isolated leak test must run on the software lane exactly
    // where `init()` would have. The loud fallback line follows
    // `devices()`' discipline, including its silence under the
    // QUANTA_CPU=1 opt-in.
    #[cfg(feature = "software")]
    if (forced == Some(ForcedBackend::Cpu) || forced.is_none())
        && let Some(dev) = driver::cpu::discover().into_iter().next()
    {
        let cpu_opt_in = std::env::var("QUANTA_CPU")
            .map(|v| v == "1")
            .unwrap_or(false);
        if forced.is_none() && !cpu_opt_in {
            std::eprintln!(
                "quanta: no GPU backend available — isolated init falling \
                 back to SOFTWARE rendering (CPU device)"
            );
        }
        return Ok(Gpu::new(maybe_validate(dev)));
    }

    Err(no_device_after_discovery(forced))
}

/// Why a forced backend cannot exist in THIS binary, if so — a missing
/// cargo feature or a platform the backend is never compiled for.
/// `None` means the backend is compiled in and its absence is a
/// machine-level matter (no driver / no device).
#[cfg(feature = "std")]
fn forced_backend_build_gap(backend: ForcedBackend) -> Option<&'static str> {
    match backend {
        ForcedBackend::Cpu => (!cfg!(feature = "software")).then_some(
            "this build does not enable the `software` feature, so the CPU \
             software device is not compiled in",
        ),
        ForcedBackend::Metal => {
            if !cfg!(feature = "metal") {
                Some("this build does not enable the `metal` feature")
            } else if !cfg!(any(target_os = "macos", target_os = "ios")) {
                Some("the metal backend only exists on Apple platforms")
            } else {
                None
            }
        }
        ForcedBackend::Vulkan => {
            if !cfg!(feature = "vulkan") {
                Some("this build does not enable the `vulkan` feature")
            } else if !cfg!(any(
                target_os = "linux",
                target_os = "android",
                target_os = "windows",
                all(feature = "vulkan-portability", target_os = "macos"),
            )) {
                Some(
                    "the vulkan backend is not compiled on this platform \
                     (macOS needs the `vulkan-portability` feature)",
                )
            } else {
                None
            }
        }
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
    // Same registry as `devices()`' CPU entries, so `init_cpu()` and a
    // `QUANTA_BACKEND=cpu` init in one process share one device.
    api::registry::get_or_discover(api::registry::BackendKind::Cpu, driver::cpu::discover)
        .into_iter()
        .next()
        .expect("cpu discover always yields one device")
}

/// Test-support sibling of [`init_cpu`]: a PRIVATE CPU software device
/// that bypasses the process-wide device registry — same contract as
/// [`init_isolated`], pinned to the software backend. For the
/// CPU-lane lifecycle/leak tests whose absolute
/// `debug_registry_counts` snapshots need a device no parallel test
/// can allocate on.
#[cfg(feature = "software")]
#[doc(hidden)]
pub fn init_cpu_isolated() -> Gpu {
    Gpu::new(maybe_validate(
        driver::cpu::discover()
            .into_iter()
            .next()
            .expect("cpu discover always yields one device"),
    ))
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
