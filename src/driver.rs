#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(any(feature = "vulkan", test))]
pub(crate) mod spirv_meta;

#[cfg(feature = "software")]
pub mod cpu;

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod webgpu;

#[cfg(feature = "std")]
pub mod validation;
