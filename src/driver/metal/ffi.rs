//! Raw Metal and Objective-C FFI bindings.
//!
//! Minimal ObjC runtime + Metal API surface required by the Quanta GPU driver.
//! No external crate dependencies — just `extern "C"` calls to the system frameworks.

#![allow(unsafe_op_in_unsafe_fn, dead_code, clippy::upper_case_acronyms)]

mod constants;
mod extern_fns;
mod structs;

// Re-export everything so existing `ffi::Foo` paths keep working.
pub use constants::*;
pub use extern_fns::*;
pub use structs::*;
