#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[cfg(feature = "vulkan")]
pub mod vulkan;

// Only the Vulkan compute path (wave creation reads the SPIR-V
// workgroup size) uses this at runtime; unit tests always compile it.
#[cfg(any(all(feature = "vulkan", feature = "compute"), test))]
pub(crate) mod spirv_meta;

#[cfg(feature = "software")]
pub mod cpu;

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod webgpu;

#[cfg(feature = "std")]
pub mod validation;
