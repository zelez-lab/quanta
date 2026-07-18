//! Shader (vertex/fragment) IR type definitions.

/// Shader pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex = 0,
    Fragment = 1,
}

/// Shader data types used in vertex/fragment parameters and return types.
///
/// Serialized by discriminant (`ty as u8`) in the wire format, so variants are
/// append-only: existing tags 0-5 must never move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    F32 = 0,
    Vec2 = 1,
    Vec3 = 2,
    Vec4 = 3,
    Mat4 = 4,
    Mat3 = 5,
    /// 32-bit unsigned integer scalar. As a vertex attribute it is an integer
    /// Input (fed by `AttributeFormat::UInt`); as a varying it must be
    /// flat-interpolated on every backend (SPIR-V `Flat` on the vertex Output
    /// AND fragment Input, MSL `[[flat]]` — integers cannot be interpolated).
    U32 = 6,
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
///
/// `metallib` is the macOS-platform Metal library. `metallib_ios` and
/// `metallib_ios_sim` are the platform-correct variants for an iOS device
/// and the iOS simulator; each is `None` when its SDK was absent at
/// compile time (a Command-Line-Tools-only mac ships macOS-only) or the
/// platform was excluded via `QUANTA_METAL_PLATFORMS`. The runtime picks
/// among them by compile target (see `ShaderBinary::for_vendor`).
#[derive(Debug, Clone)]
pub struct ShaderOutput {
    pub spirv: Option<Vec<u8>>,
    pub metallib: Option<Vec<u8>>,
    pub metallib_ios: Option<Vec<u8>>,
    pub metallib_ios_sim: Option<Vec<u8>>,
    pub wgsl: Option<String>,
}
