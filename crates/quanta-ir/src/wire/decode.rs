//! Binary decoding: Reader + all read_* helpers.

mod header;
mod helpers;
mod ops;

pub(crate) use header::{
    Reader, read_compiler_output, read_kernel_def, read_shader_def, read_shader_output,
};
