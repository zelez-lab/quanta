//! KernelDef → WebGPU Shading Language.

mod helpers;
mod kernel;
mod ops;

pub use helpers::{emit_fragment_shader, emit_vertex_shader};
pub use kernel::emit;
