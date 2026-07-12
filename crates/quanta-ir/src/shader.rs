//! Shader (vertex/fragment) IR type definitions.

/// Shader pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex = 0,
    Fragment = 1,
}

/// Shader data types used in vertex/fragment parameters and return types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    F32 = 0,
    Vec2 = 1,
    Vec3 = 2,
    Vec4 = 3,
    Mat4 = 4,
    Mat3 = 5,
}

/// A parsed shader parameter (vertex attribute, uniform, or slice binding).
///
/// `is_uniform` marks a `&T` uniform; `is_slice` marks a `&[T]` storage-buffer
/// array (`ty` is then the element type). The two are mutually exclusive, and a
/// param with neither set is a plain value attribute. Uniform and slice params
/// share one binding space (see the compiler's shared decl-index): the runtime
/// binds both with `.uniform(slot, …)` as a storage-buffer descriptor at
/// binding=slot on both stages.
#[derive(Debug, Clone)]
pub struct ShaderParam {
    pub name: String,
    pub ty: ShaderType,
    pub is_uniform: bool,
    pub is_slice: bool,
}

/// Complete shader definition — input to the compiler for vertex/fragment shaders.
#[derive(Debug, Clone)]
pub struct ShaderDef {
    pub name: String,
    pub stage: ShaderStage,
    pub params: Vec<ShaderParam>,
    pub return_type: ShaderType,
    pub body_source: String,
}

/// Compiler output for shader stages — SPIR-V and metallib binaries.
#[derive(Debug, Clone)]
pub struct ShaderOutput {
    pub spirv: Option<Vec<u8>>,
    pub metallib: Option<Vec<u8>>,
    pub wgsl: Option<String>,
}
