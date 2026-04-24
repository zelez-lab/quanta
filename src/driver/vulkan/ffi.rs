//! Raw Vulkan FFI bindings — minimal subset for GPU compute and rendering.
//!
//! Follows Dija's pattern: opaque handles, `#[repr(C)]` structs, platform-gated
//! extern blocks. No `ash` dependency.

#![allow(non_camel_case_types, dead_code)]

pub mod constants;
pub mod device;
pub mod extern_fns;
pub mod structs;
pub mod structs_render;

pub use constants::*;
pub use device::*;
pub use extern_fns::*;
pub use structs::*;
pub use structs_render::*;
