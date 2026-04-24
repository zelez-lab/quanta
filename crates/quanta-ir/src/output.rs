//! Compiler output types.

/// Compiler output — compiled kernel for all targets.
#[derive(Debug, Clone)]
pub struct CompilerOutput {
    pub amd: Option<Vec<u8>>,
    pub nvidia: Option<Vec<u8>>,
    pub spirv: Option<Vec<u8>>,
    pub metallib: Option<Vec<u8>>,
    pub wgsl: Option<String>,
}
