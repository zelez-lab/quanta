#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal;

// The Vulkan FEATURE is target-independent (it is on by default), but the
// Vulkan CODE is `cfg(target_os)`-gated so a plain Apple build compiles no
// Vulkan and needs no libvulkan (the MoltenVK link trap). Active on
// Linux / Android / Windows, and on macOS only under `vulkan-portability`
// (MoltenVK). Mirrors how `metal` is gated to any(macos, ios) above.
#[cfg(all(
    feature = "vulkan",
    any(
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        all(feature = "vulkan-portability", target_os = "macos"),
    )
))]
pub mod vulkan;

// The Vulkan compute path (wave creation reads the SPIR-V workgroup
// size) and the render path (pipeline creation reflects a fragment's
// color-output count) both use this at runtime; unit tests always
// compile it. The Vulkan arm carries the same target_os gate as the
// module so a pruned Vulkan build drops it too; the `render` arm keeps
// it compiled on every OS (including Apple) for the pipeline path.
#[cfg(any(
    all(
        feature = "vulkan",
        feature = "compute",
        any(
            target_os = "linux",
            target_os = "android",
            target_os = "windows",
            all(feature = "vulkan-portability", target_os = "macos"),
        )
    ),
    feature = "render",
    test
))]
pub(crate) mod spirv_meta;

#[cfg(feature = "software")]
pub mod cpu;

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod webgpu;

#[cfg(feature = "std")]
pub mod validation;

/// Weak back-reference to the shared `Arc<dyn GpuDevice>` a driver is
/// held through, installed once via `GpuDevice::install_self_ref`.
/// Drivers upgrade it when minting a `Pulse` so every pulse keeps its
/// device alive. Weak here, strong in the pulse — no cycle: the device
/// never holds pulses. Only the async GPU backends carry one — the
/// synchronous backends' pulses defer no device work — so it is gated
/// like the metal/vulkan modules above.
#[cfg(any(
    all(feature = "metal", any(target_os = "macos", target_os = "ios")),
    all(
        feature = "vulkan",
        any(
            target_os = "linux",
            target_os = "android",
            target_os = "windows",
            all(feature = "vulkan-portability", target_os = "macos"),
        )
    ),
))]
pub(crate) struct DeviceSelfRef(std::sync::OnceLock<alloc::sync::Weak<dyn crate::GpuDevice>>);

#[cfg(any(
    all(feature = "metal", any(target_os = "macos", target_os = "ios")),
    all(
        feature = "vulkan",
        any(
            target_os = "linux",
            target_os = "android",
            target_os = "windows",
            all(feature = "vulkan-portability", target_os = "macos"),
        )
    ),
))]
impl DeviceSelfRef {
    pub(crate) fn new() -> Self {
        Self(std::sync::OnceLock::new())
    }

    /// First install wins; a later call is a no-op (there is exactly
    /// one shared `Arc` per device).
    pub(crate) fn install(&self, self_ref: alloc::sync::Weak<dyn crate::GpuDevice>) {
        let _ = self.0.set(self_ref);
    }

    /// The strong keep-alive a freshly minted pulse carries. `None`
    /// only before `install_self_ref` ran (a device not yet handed to
    /// callers — nothing can outlive it) or during device teardown.
    /// Metal's only caller is compute/render-gated (`make_async_pulse`),
    /// so a pruned metal-only build compiles this without a caller.
    #[cfg_attr(not(any(feature = "compute", feature = "render")), allow(dead_code))]
    pub(crate) fn pulse_keep_alive(&self) -> Option<alloc::sync::Arc<dyn crate::GpuDevice>> {
        self.0.get().and_then(|w| w.upgrade())
    }
}
