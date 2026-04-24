//! Call the quanta-compiler binary or use built-in emitters.
//!
//! The MSL/WGSL emitter code is kept for Phase 4 (JIT migration to quanta-ir).

#[allow(unused_imports)]
mod binary;
mod emit_msl;
mod emit_wgsl;
pub(crate) mod shader_emit;
pub(crate) mod shader_types;

pub use binary::compile_kernel;
#[allow(unused_imports)]
pub(crate) use binary::{ShaderCompileOutput, compile_shader};
#[allow(unused_imports)]
pub(crate) use shader_types::{
    ShaderParam, ShaderType, extract_body_source, parse_return_type, parse_shader_params,
};
