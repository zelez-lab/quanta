//! Metal driver for macOS/iOS.
//!
//! Uses raw ObjC/Metal FFI bindings — no external Metal crate dependency.
//! Covers compute dispatch, render pass execution, texture management,
//! depth/stencil, instanced/indexed/indirect draw, MRT, and debug labels.

mod compute;
mod device;
mod device_impl;
pub(crate) mod ffi;
mod memory;
#[cfg(feature = "render")]
mod render;
mod sparse;
mod texture;

// Re-export public API from submodules.
pub use device::{MetalDevice, discover};
pub(crate) use device_impl::*;
