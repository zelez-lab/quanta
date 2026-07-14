//! Shared internals behind the Quanta DSL face crates.
//!
//! Shared by the two DSL face crates (`quanta-compute-dsl` and
//! `quanta-render-dsl`); not a public API. The items re-exported below are
//! `pub` only so the face crates can call them across the crate boundary —
//! their surface is the `#[quanta::*]` macros, not this crate.
//!
//! Contents:
//! - compiler-binary discovery, the rev handshake, and invocation
//!   (`binary`): `compile_kernel` for the compute face, `compile_shader`
//!   for the render face;
//! - shader parameter parsing and body extraction (`shader_types`).
//!
//! The crate compiles whole and featureless: both faces' entry points are
//! plain functions, and each face crate calls only the ones it needs.

#[allow(unused_imports)]
mod binary;
pub mod shader_types;

pub use binary::{ShaderCompileOutput, compile_kernel, compile_shader};
#[allow(unused_imports)]
pub use shader_types::{
    ShaderParam, ShaderType, extract_body_source, parse_return_type, parse_shader_params,
    rewrite_texture_names,
};
