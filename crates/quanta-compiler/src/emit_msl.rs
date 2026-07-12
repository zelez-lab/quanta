//! KernelDef → Metal Shading Language.
//!
//! Walks KernelOps and emits correct MSL for all supported operations.
//! This is the structured emitter — no string replacement.

mod helpers;
mod kernel;
mod ops;
mod shader;
mod shader_ast;

pub use kernel::emit;
pub use shader::{emit_fragment_shader, emit_vertex_shader};
