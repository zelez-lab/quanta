//! Quanta Conformance Test Suite
//!
//! Every driver (Metal, Vulkan, Software, Zelez) must pass all tests.
//! Tests take a `&Gpu` — the driver is transparent.
//!
//! Run: cargo test --test conformance

pub mod compute;
pub mod memory;
pub mod texture;
