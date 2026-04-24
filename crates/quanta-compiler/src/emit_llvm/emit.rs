//! Kernel building and op emission.
//!
//! Split into submodules:
//! - `context` — build_kernel entry point, parameter mapping, register setup
//! - `ops`     — emit_ops / emit_op (KernelOp dispatch match)
//! - `helpers` — emit_binop, emit_cmp, emit_unary, emit_cast, emit_math_direct, make_vec_type

mod context;
mod helpers;
mod ops;

pub(crate) use context::build_kernel;
