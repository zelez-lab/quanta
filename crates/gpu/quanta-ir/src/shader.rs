//! Shader (vertex/fragment) IR type definitions.

/// Shader pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    /// Vertex stage — consumes attributes, returns the varyings struct.
    Vertex = 0,
    /// Fragment stage — consumes the varyings struct, returns a colour.
    Fragment = 1,
}

/// Shader data types used in vertex/fragment parameters and return types.
///
/// Serialized by discriminant (`ty as u8`) in the wire format, so variants are
/// append-only: existing tags 0-5 must never move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    /// Single-precision scalar.
    F32 = 0,
    /// Two-component float vector.
    Vec2 = 1,
    /// Three-component float vector.
    Vec3 = 2,
    /// Four-component float vector — also the type of the `#[position]`
    /// varying.
    Vec4 = 3,
    /// 4×4 float matrix.
    Mat4 = 4,
    /// 3×3 float matrix.
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
///
/// Plain value params are VERTEX-only (vertex attributes). Fragment stage
/// inputs come from the shader's [`ShaderVaryings`] interface; a fragment
/// `ShaderDef` carrying a plain value param is rejected by every emitter.
#[derive(Debug, Clone)]
pub struct ShaderParam {
    /// Parameter name as written in the source.
    pub name: String,
    /// The parameter's type, or the element type when `is_slice` is set.
    pub ty: ShaderType,
    /// Whether the parameter is a `&T` uniform.
    pub is_uniform: bool,
    /// Whether the parameter is a `&[T]` storage-buffer array.
    pub is_slice: bool,
}

/// One named varying of a vertex↔fragment interface struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaryingField {
    /// Field name, which both stages spell to reach the varying.
    pub name: String,
    /// The varying's type.
    pub ty: ShaderType,
}

/// The vertex↔fragment interface under the shared-struct model (the
/// WGSL/HLSL-convergent design): one user struct, derived with
/// `#[derive(quanta::Varyings)]`, is the single explicit interface between
/// the two stages. The vertex RETURNS it (a struct literal in tail
/// position); the fragment TAKES it as its single stage-input param and
/// reads varyings by field name.
///
/// - `position` names the `#[position]`-marked field (always a `Vec4`): the
///   vertex routes it to gl_Position / `[[position]]`; a fragment reading it
///   sees the interpolated window position (FragCoord semantics, as in WGSL).
/// - `fields` are the non-position varyings in field-DECLARATION order:
///   field `i` is Location `i` on every backend, deterministically. Integer
///   (`U32`) fields are flat-interpolated on both ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderVaryings {
    /// The struct's type name as written in the source (`Surface`). The
    /// vertex body's tail literal names it; the MSL/WGSL emitters reuse it
    /// as the interface struct's name.
    pub struct_name: String,
    /// The `#[position]` field's name (type `Vec4` — gl_Position).
    pub position: String,
    /// The non-position varyings in declaration order — field `i` is
    /// Location `i`.
    pub fields: Vec<VaryingField>,
    /// Fragment stage: the receiving parameter's name (`s` in
    /// `fn fs(s: Surface)`), which the body's `s.<field>` accesses resolve
    /// against. `None` on the vertex stage (its body names the struct in the
    /// tail literal instead).
    pub binding: Option<String>,
}

impl ShaderVaryings {
    /// The declared type of a non-position varying, by field name.
    pub fn field_type(&self, name: &str) -> Option<ShaderType> {
        self.fields.iter().find(|f| f.name == name).map(|f| f.ty)
    }
}

/// Complete shader definition — input to the compiler for vertex/fragment shaders.
///
/// `varyings` carries the vertex↔fragment interface under the shared-struct
/// model. `None` means the stage has NO varyings: a vertex returning a bare
/// `Vec4` (position-only), or a fragment with no stage inputs (uniforms /
/// textures / `frag_coord()` only).
#[derive(Debug, Clone)]
pub struct ShaderDef {
    /// Shader name, which becomes the entry point's name.
    pub name: String,
    /// Which pipeline stage this shader is.
    pub stage: ShaderStage,
    /// Attributes, uniforms and slice bindings the shader takes.
    pub params: Vec<ShaderParam>,
    /// The stage's return type.
    pub return_type: ShaderType,
    /// Raw Rust source of the body, which the emitters translate.
    pub body_source: String,
    /// The vertex↔fragment interface, or `None` when the stage has none.
    pub varyings: Option<ShaderVaryings>,
}

/// Compiler output for shader stages — SPIR-V and metallib binaries.
///
/// `metallib` is the macOS-platform Metal library. `metallib_ios` and
/// `metallib_ios_sim` are the platform-correct variants for an iOS device
/// and the iOS simulator; each is `None` when its SDK was absent at
/// compile time (a Command-Line-Tools-only mac ships macOS-only) or the
/// platform was excluded via `QUANTA_METAL_PLATFORMS`. The runtime picks
/// among them by compile target (see `ShaderBinary::for_artifact`).
#[derive(Debug, Clone)]
pub struct ShaderOutput {
    /// SPIR-V module for the Vulkan driver.
    pub spirv: Option<Vec<u8>>,
    /// Metal library built for macOS.
    pub metallib: Option<Vec<u8>>,
    /// Metal library built for an iOS device.
    pub metallib_ios: Option<Vec<u8>>,
    /// Metal library built for the iOS simulator.
    pub metallib_ios_sim: Option<Vec<u8>>,
    /// WGSL source for the WebGPU driver.
    pub wgsl: Option<String>,
}
