//! Vulkan driver for Linux, Android, Windows, and macOS (via MoltenVK).
//!
//! Uses raw FFI bindings (no `ash` dependency).
//! Covers compute dispatch, render pass execution, texture management,
//! depth/stencil, instanced/indexed/indirect draw, MRT, and debug labels.

mod compute;
mod device;
mod device_impl;
pub(crate) mod ffi;
mod helpers;
mod memory;
mod render;
mod sync;
mod texture;

// Re-export public API.
pub use device::{VulkanDevice, discover};

// Re-export internal types used by submodules via `super::`.
pub(self) use device::{VkBuffer, VkComputePipeline, VkQueryPool, VkRenderPipeline, VkTexture};
pub(self) use helpers::{
    address_to_vk, blend_factor_to_vk, blend_op_to_vk, compare_op_to_vk, filter_to_vk,
    format_bytes_per_pixel_vk, format_to_vulkan, sample_count_to_vk,
};
