//! Quanta kernel intermediate representation.
//!
//! Shared between the proc macro (`quanta-macros`) and the compiler binary
//! (`quanta-compiler`). Defines the platform-agnostic IR that represents
//! GPU kernels between parsing and code generation.

pub mod wire;

/// SPIR-V emitter for JIT compilation. Enabled by the `jit` feature.
#[cfg(feature = "jit")]
pub mod emit_spirv;

/// MSL emitter for JIT compilation. Enabled by the `jit` feature.
#[cfg(feature = "jit")]
pub mod emit_msl;

/// Scalar types supported in GPU kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    F16,
    F32,
    F64,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Bool,
}

/// Virtual register (SSA-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

/// Constant value.
#[derive(Debug, Clone, Copy)]
pub enum ConstValue {
    F16(u16),
    F32(f32),
    F64(f64),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    Bool(bool),
}

/// Kernel parameter — how function arguments map to GPU bindings.
#[derive(Debug, Clone)]
pub enum KernelParam {
    FieldRead {
        name: String,
        slot: u32,
        scalar_type: ScalarType,
    },
    FieldWrite {
        name: String,
        slot: u32,
        scalar_type: ScalarType,
    },
    Constant {
        name: String,
        slot: u32,
        scalar_type: ScalarType,
    },
    Texture2DRead {
        name: String,
        slot: u32,
        scalar_type: ScalarType,
    },
    Texture2DWrite {
        name: String,
        slot: u32,
        scalar_type: ScalarType,
    },
    Texture3DRead {
        name: String,
        slot: u32,
        scalar_type: ScalarType,
    },
}

/// Binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    SatAdd,
    SatSub,
}

/// Unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    BitNot,
    LogicalNot,
}

/// Comparison operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    Sub,
    Min,
    Max,
    And,
    Or,
    Xor,
    Exchange,
    CompareExchange,
}

/// Built-in math functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathFn {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sqrt,
    Rsqrt,
    Exp,
    Exp2,
    Log,
    Log2,
    Pow,
    Abs,
    Min,
    Max,
    Clamp,
    Floor,
    Ceil,
    Round,
    Fma,
}

/// A single kernel IR operation.
#[derive(Debug, Clone)]
pub enum KernelOp {
    // Memory
    Load {
        dst: Reg,
        field: u32,
        index: Reg,
        ty: ScalarType,
    },
    Store {
        field: u32,
        index: Reg,
        src: Reg,
        ty: ScalarType,
    },
    SharedDecl {
        id: u32,
        ty: ScalarType,
        count: u32,
    },
    SharedLoad {
        dst: Reg,
        id: u32,
        index: Reg,
        ty: ScalarType,
    },
    SharedStore {
        id: u32,
        index: Reg,
        src: Reg,
        ty: ScalarType,
    },

    // Arithmetic
    BinOp {
        dst: Reg,
        a: Reg,
        b: Reg,
        op: BinOp,
        ty: ScalarType,
    },
    UnaryOp {
        dst: Reg,
        a: Reg,
        op: UnaryOp,
        ty: ScalarType,
    },
    Cmp {
        dst: Reg,
        a: Reg,
        b: Reg,
        op: CmpOp,
        ty: ScalarType,
    },

    // Control flow
    Branch {
        cond: Reg,
        then_ops: Vec<KernelOp>,
        else_ops: Vec<KernelOp>,
    },
    Loop {
        count: Reg,
        iter_reg: Reg,
        body: Vec<KernelOp>,
    },

    // Math
    MathCall {
        dst: Reg,
        func: MathFn,
        args: Vec<Reg>,
        ty: ScalarType,
    },

    // Thread indexing
    QuarkId {
        dst: Reg,
    },
    QuarkCount {
        dst: Reg,
    },
    LocalId {
        dst: Reg,
    },
    GroupId {
        dst: Reg,
    },
    GroupSize {
        dst: Reg,
    },

    // Synchronization
    Barrier,
    AtomicOp {
        dst: Reg,
        field: u32,
        index: Reg,
        val: Reg,
        op: AtomicOp,
        ty: ScalarType,
    },
    AtomicCas {
        dst: Reg,
        field: u32,
        index: Reg,
        expected: Reg,
        desired: Reg,
        ty: ScalarType,
    },

    // Warp/wave
    WaveShuffle {
        dst: Reg,
        src: Reg,
        lane_delta: Reg,
        ty: ScalarType,
    },
    WaveBallot {
        dst: Reg,
        predicate: Reg,
    },
    WaveAny {
        dst: Reg,
        predicate: Reg,
    },
    WaveAll {
        dst: Reg,
        predicate: Reg,
    },

    // Type conversion
    Cast {
        dst: Reg,
        src: Reg,
        from: ScalarType,
        to: ScalarType,
    },
    Const {
        dst: Reg,
        value: ConstValue,
    },

    // Vector
    VecConstruct {
        dst: Reg,
        components: Vec<Reg>,
        ty: ScalarType,
    },
    VecExtract {
        dst: Reg,
        vec: Reg,
        component: u8,
        ty: ScalarType,
    },
    MatMul {
        dst: Reg,
        a: Reg,
        b: Reg,
        size: u8,
        ty: ScalarType,
    },
    /// Cooperative matrix multiply-accumulate (tensor cores / SIMD group matrix).
    /// D = A * B + C where A, B, C, D are SIMD-group-scoped matrices.
    CooperativeMMA {
        dst: Reg,
        a: Reg,
        b: Reg,
        c: Reg,
        m: u8,
        n: u8,
        k: u8,
        ty: ScalarType,
    },

    // Texture
    TextureSample2D {
        dst: Reg,
        texture: u32,
        x: Reg,
        y: Reg,
        ty: ScalarType,
    },
    TextureSample3D {
        dst: Reg,
        texture: u32,
        x: Reg,
        y: Reg,
        z: Reg,
        ty: ScalarType,
    },
    TextureWrite2D {
        texture: u32,
        x: Reg,
        y: Reg,
        value: Reg,
        ty: ScalarType,
    },
    TextureSize {
        dst_w: Reg,
        dst_h: Reg,
        texture: u32,
    },

    // Register copy (for loop-carried variable updates)
    Copy {
        dst: Reg,
        src: Reg,
        ty: ScalarType,
    },

    // Control flow
    Break,

    // Dynamic parallelism
    Dispatch {
        wave: Reg,
        groups: [Reg; 3],
    },

    // Device function call (user-defined helper)
    DeviceCall {
        dst: Reg,
        func_name: String,
        args: Vec<Reg>,
        ty: ScalarType,
    },

    // Bit manipulation
    Bitcast {
        dst: Reg,
        src: Reg,
        from: ScalarType,
        to: ScalarType,
    },
    CountTrailingZeros {
        dst: Reg,
        src: Reg,
        ty: ScalarType,
    },
    CountLeadingZeros {
        dst: Reg,
        src: Reg,
        ty: ScalarType,
    },
    PopCount {
        dst: Reg,
        src: Reg,
        ty: ScalarType,
    },

    // Dot product (vector)
    Dot {
        dst: Reg,
        a: Reg,
        b: Reg,
        ty: ScalarType,
        width: u8,
    },

    // Subgroup
    SubgroupSize {
        dst: Reg,
    },

    // Subgroup scan/reduce
    SubgroupReduceAdd {
        dst: Reg,
        src: Reg,
        ty: ScalarType,
    },
    SubgroupReduceMin {
        dst: Reg,
        src: Reg,
        ty: ScalarType,
    },
    SubgroupReduceMax {
        dst: Reg,
        src: Reg,
        ty: ScalarType,
    },
    SubgroupExclusiveAdd {
        dst: Reg,
        src: Reg,
        ty: ScalarType,
    },
    SubgroupInclusiveAdd {
        dst: Reg,
        src: Reg,
        ty: ScalarType,
    },

    // Texture load without sampler
    TextureLoad2D {
        dst: Reg,
        texture: u32,
        x: Reg,
        y: Reg,
        ty: ScalarType,
    },

    // Dynamic shared memory declaration (size determined at dispatch)
    SharedDeclDyn {
        id: u32,
        ty: ScalarType,
    },

    // GPU debug print (writes value + thread_id to a debug buffer)
    DebugPrint {
        src: Reg,
        ty: ScalarType,
    },
}

/// A device function definition — parsed inner `fn` callable from a kernel.
///
/// Contains the function's name, parameter types, return type, and body as
/// KernelOps. Used by the SPIR-V emitter to generate proper `OpFunction`/
/// `OpFunctionCall` instructions.
#[derive(Debug, Clone)]
pub struct DeviceFnDef {
    pub name: String,
    pub params: Vec<(String, ScalarType)>,
    pub return_type: ScalarType,
    pub body: Vec<KernelOp>,
    pub next_reg: u32,
}

/// Complete kernel definition in IR form.
#[derive(Debug, Clone)]
pub struct KernelDef {
    pub name: String,
    pub params: Vec<KernelParam>,
    pub body: Vec<KernelOp>,
    /// Raw Rust source of the body (temporary — used for string-based MSL/WGSL
    /// emission until Phase 2 populates `body` with real KernelOps).
    pub body_source: Option<String>,
    pub next_reg: u32,
    /// Optimization level: 0 (none), 1, 2, 3 (aggressive). Default: 3.
    pub opt_level: u8,
    /// Source text of `#[quanta::device]` helper functions used by this kernel.
    /// Each entry is the original Rust source of an inner `fn` defined in the
    /// kernel body. The MSL/WGSL emitters prepend these as GPU helper functions;
    /// the rustc compilation path includes them in the generated crate.
    pub device_sources: Vec<String>,
    /// Parsed device function definitions with KernelOp bodies.
    /// Populated by the parser for all inner `fn` definitions.
    /// The SPIR-V emitter uses these to generate real function calls.
    pub device_functions: Vec<DeviceFnDef>,
    /// Workgroup (threadgroup) size: [x, y, z]. Default: [64, 1, 1].
    /// Set via `#[quanta::kernel(workgroup = [256])]` or similar.
    pub workgroup_size: [u32; 3],
    /// Required subgroup (warp/simd) size. None = use hardware default.
    /// Set via `#[quanta::kernel(subgroup = 32)]`.
    pub subgroup_size: Option<u32>,
    /// Dynamic shared memory size in bytes, set by dispatch API. 0 = none.
    pub dynamic_shared_bytes: u32,
}

/// Compiler output — compiled kernel for all targets.
#[derive(Debug, Clone)]
pub struct CompilerOutput {
    pub amd: Option<Vec<u8>>,
    pub nvidia: Option<Vec<u8>>,
    pub spirv: Option<Vec<u8>>,
    pub metallib: Option<Vec<u8>>,
    pub wgsl: Option<String>,
}

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

impl ScalarType {
    /// Metal Shading Language type name.
    pub fn msl_name(&self) -> &'static str {
        match self {
            Self::F16 => "half",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::U8 => "uint8_t",
            Self::U16 => "ushort",
            Self::U32 => "uint",
            Self::U64 => "ulong",
            Self::I8 => "int8_t",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "long",
            Self::Bool => "bool",
        }
    }

    /// WebGPU Shading Language type name.
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U8 | Self::U16 | Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I8 | Self::I16 | Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Bool => "bool",
        }
    }
}

// ── Shader (vertex/fragment) types ─────────────────────────────────────────

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

/// A parsed shader parameter (vertex attribute or uniform binding).
#[derive(Debug, Clone)]
pub struct ShaderParam {
    pub name: String,
    pub ty: ShaderType,
    pub is_uniform: bool,
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
