//! Custom binary serialization for quanta-ir types.
//!
//! Replaces serde + bincode with a zero-dependency, no_std-compatible wire
//! format. All integers are little-endian. Strings and byte slices are
//! length-prefixed (u32). Options use a u8 tag (0 = None, 1 = Some). Enums
//! use a u8 discriminant followed by variant-specific fields.

mod decode;
mod encode;

#[cfg(test)]
#[path = "wire/tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "wire/kani_proofs.rs"]
mod kani_proofs;

use crate::*;

use decode::{Reader, read_compiler_output, read_kernel_def, read_shader_def, read_shader_output};
use encode::{
    Writer, write_compiler_output, write_kernel_def, write_shader_def, write_shader_output,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Serialize a [`KernelDef`] to wire bytes.
pub fn serialize_kernel(kernel: &KernelDef) -> Vec<u8> {
    let mut w = Writer::with_capacity(256);
    write_kernel_def(&mut w, kernel);
    w.finish()
}

/// Deserialize a [`KernelDef`] from wire bytes.
pub fn deserialize_kernel(bytes: &[u8]) -> Result<KernelDef, &'static str> {
    let mut r = Reader::new(bytes);
    let k = read_kernel_def(&mut r)?;
    if r.remaining() != 0 {
        return Err("trailing bytes after KernelDef");
    }
    Ok(k)
}

/// Serialize a [`CompilerOutput`] to wire bytes.
pub fn serialize_output(output: &CompilerOutput) -> Vec<u8> {
    let mut w = Writer::with_capacity(256);
    write_compiler_output(&mut w, output);
    w.finish()
}

/// Deserialize a [`CompilerOutput`] from wire bytes.
pub fn deserialize_output(bytes: &[u8]) -> Result<CompilerOutput, &'static str> {
    let mut r = Reader::new(bytes);
    let o = read_compiler_output(&mut r)?;
    if r.remaining() != 0 {
        return Err("trailing bytes after CompilerOutput");
    }
    Ok(o)
}

/// Serialize a [`ShaderDef`] to wire bytes.
pub fn serialize_shader(shader: &ShaderDef) -> Vec<u8> {
    let mut w = Writer::with_capacity(256);
    write_shader_def(&mut w, shader);
    w.finish()
}

/// Deserialize a [`ShaderDef`] from wire bytes.
pub fn deserialize_shader(bytes: &[u8]) -> Result<ShaderDef, &'static str> {
    let mut r = Reader::new(bytes);
    let s = read_shader_def(&mut r)?;
    if r.remaining() != 0 {
        return Err("trailing bytes after ShaderDef");
    }
    Ok(s)
}

/// Serialize a [`ShaderOutput`] to wire bytes.
pub fn serialize_shader_output(output: &ShaderOutput) -> Vec<u8> {
    let mut w = Writer::with_capacity(256);
    write_shader_output(&mut w, output);
    w.finish()
}

/// Deserialize a [`ShaderOutput`] from wire bytes.
pub fn deserialize_shader_output(bytes: &[u8]) -> Result<ShaderOutput, &'static str> {
    let mut r = Reader::new(bytes);
    let o = read_shader_output(&mut r)?;
    if r.remaining() != 0 {
        return Err("trailing bytes after ShaderOutput");
    }
    Ok(o)
}
