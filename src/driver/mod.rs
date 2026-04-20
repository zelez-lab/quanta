#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "naga-shaders")]
#[allow(dead_code)]
pub mod shader_convert;

#[cfg(feature = "vulkan")]
pub mod spirv;
#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "software")]
pub mod software;

#[cfg(feature = "std")]
pub mod validation;
