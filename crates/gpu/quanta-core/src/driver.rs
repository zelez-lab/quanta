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
