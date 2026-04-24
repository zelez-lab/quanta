//! Binary encoding: Writer + all write_* helpers.

mod header;
mod helpers;
mod ops;

pub(crate) use header::{
    Writer, write_compiler_output, write_kernel_def, write_shader_def, write_shader_output,
};
