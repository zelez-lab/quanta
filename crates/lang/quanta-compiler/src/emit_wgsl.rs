//! KernelDef → WebGPU Shading Language.
//!
//! This is now a thin re-export of `quanta_ir::emit_wgsl`. The single source
//! of truth lives in `quanta-ir` because the same emitter must run inside
//! the wasm binary served to browsers (JIT path, step 050) and at build
//! time from this compiler binary.

pub use quanta_ir::emit_wgsl::{emit, emit_fragment_shader, emit_vertex_shader};
