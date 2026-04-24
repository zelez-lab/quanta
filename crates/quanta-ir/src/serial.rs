//! Serialization helpers for IR types.

use crate::output::CompilerOutput;
use crate::shader::{ShaderDef, ShaderOutput};
use crate::types::KernelDef;
use crate::wire;

/// Serialize a KernelDef to binary bytes.
pub fn serialize_kernel(kernel: &KernelDef) -> Vec<u8> {
    wire::serialize_kernel(kernel)
}

/// Deserialize a KernelDef from binary bytes.
pub fn deserialize_kernel(bytes: &[u8]) -> Result<KernelDef, &'static str> {
    wire::deserialize_kernel(bytes)
}

/// Serialize a CompilerOutput to binary bytes.
pub fn serialize_output(output: &CompilerOutput) -> Vec<u8> {
    wire::serialize_output(output)
}

/// Deserialize a CompilerOutput from binary bytes.
pub fn deserialize_output(bytes: &[u8]) -> Result<CompilerOutput, &'static str> {
    wire::deserialize_output(bytes)
}

/// Serialize a [`ShaderDef`] to wire bytes.
pub fn serialize_shader(shader: &ShaderDef) -> Vec<u8> {
    wire::serialize_shader(shader)
}

/// Deserialize a [`ShaderDef`] from wire bytes.
pub fn deserialize_shader(bytes: &[u8]) -> Result<ShaderDef, &'static str> {
    wire::deserialize_shader(bytes)
}

/// Serialize a [`ShaderOutput`] to wire bytes.
pub fn serialize_shader_output(output: &ShaderOutput) -> Vec<u8> {
    wire::serialize_shader_output(output)
}

/// Deserialize a [`ShaderOutput`] from wire bytes.
pub fn deserialize_shader_output(bytes: &[u8]) -> Result<ShaderOutput, &'static str> {
    wire::deserialize_shader_output(bytes)
}
