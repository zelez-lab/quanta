//! The process-wide device registry — one live [`DeviceContext`] per
//! physical device.
//!
//! `init()` / `devices()` used to run full discovery per call: a fresh
//! `VkInstance` + `VkDevice` (and their Metal/CPU equivalents) every
//! time. Test suites calling `init()` per test on parallel threads
//! stormed real drivers into `VK_ERROR_INCOMPATIBLE_DRIVER`. The
//! registry makes repeated initialization converge on the SAME device:
//! callers get `Gpu` clones of one shared [`DeviceContext`], which is
//! also what the per-device singleton contracts require — one lane per
//! device means two `init()` callers must share the lane, or their
//! dispatches would commit in arbitrary order.
//!
//! Entries are `Weak`: the registry never keeps a device alive. When
//! the last `Gpu` clone drops, the device tears down exactly as
//! before, and the next `init()` rebuilds it. Keys are
//! `(backend, index-within-backend)` — discovery order is a documented
//! stable contract, so the index is an identity. Hotplug that changes
//! enumeration mid-process is out of scope (pre-1.0, documented).
//!
//! Validation wrapping (`QUANTA_VALIDATE`) is applied when a device is
//! BUILT and rides the cache after that: toggling the env var
//! mid-process does not re-wrap a live device.
//!
//! WebGPU never passes through here — its async handshake has its own
//! entry point and wasm has no parallel-test storms to absorb.

use alloc::boxed::Box;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use std::sync::Mutex;

use crate::GpuDevice;
use crate::api::gpu::{DeviceContext, Gpu};

/// Which driver produced a device. Part of the registry key, so
/// `QUANTA_BACKEND` forcing composes for free: forcing only decides
/// which backends are PROBED; identity within a backend is untouched.
// The whole registry is native-only: wasm32 builds the WebGPU driver
// alone, and that never registers (see the module docs).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendKind {
    #[cfg_attr(
        not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    Metal,
    // Constructed only where `devices()`' vulkan block compiles —
    // feature AND target; `--features vulkan` on a plain Mac build
    // enables the feature with no constructor site.
    #[cfg_attr(
        not(all(
            feature = "vulkan",
            any(
                target_os = "linux",
                target_os = "android",
                target_os = "windows",
                all(feature = "vulkan-portability", target_os = "macos"),
            )
        )),
        allow(dead_code)
    )]
    Vulkan,
    #[cfg_attr(not(feature = "software"), allow(dead_code))]
    Cpu,
}

/// One registry row: the key — backend plus index in that backend's
/// stable discovery order — and a weak ref to the live context.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
type Entry = ((BackendKind, usize), Weak<DeviceContext>);

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
static REGISTRY: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// Return this backend's devices, reusing live contexts and running
/// `discover` only when something is missing.
///
/// Fast path: every cached entry for `kind` is alive → clone them out,
/// no discovery at all (the parallel-`init()` case — no driver
/// traffic). Slow path: run `discover`, and per index prefer a still-
/// live cached context over the freshly built device (the fresh one
/// drops immediately; rare, and cheap now that the Vulkan instance is
/// process-shared). The lock is held across `discover` on purpose —
/// two racing initializations must not both build the same device.
// Called from `devices()`, `init_cpu()` — native entry points only.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn get_or_discover(
    kind: BackendKind,
    discover: impl FnOnce() -> Vec<Box<dyn GpuDevice>>,
) -> Vec<Gpu> {
    let mut reg = REGISTRY.lock().expect("device registry poisoned");

    let cached: Vec<Option<Arc<DeviceContext>>> = {
        let mut per_index: Vec<Option<Arc<DeviceContext>>> = Vec::new();
        for ((k, idx), weak) in reg.iter() {
            if *k == kind {
                if per_index.len() <= *idx {
                    per_index.resize(*idx + 1, None);
                }
                per_index[*idx] = weak.upgrade();
            }
        }
        per_index
    };
    if !cached.is_empty() && cached.iter().all(Option::is_some) {
        return cached
            .into_iter()
            .map(|ctx| Gpu {
                ctx: ctx.expect("checked all alive"),
            })
            .collect();
    }

    let fresh = discover();
    reg.retain(|((k, _), _)| *k != kind);
    let mut out = Vec::with_capacity(fresh.len());
    for (idx, dev) in fresh.into_iter().enumerate() {
        let ctx = match cached.get(idx).and_then(Clone::clone) {
            // A live context wins over the rebuild: callers holding it
            // keep their device, and the fresh duplicate drops here.
            Some(live) => live,
            None => Gpu::new(crate::maybe_validate(dev)).ctx,
        };
        reg.push(((kind, idx), Arc::downgrade(&ctx)));
        out.push(Gpu { ctx });
    }
    out
}
