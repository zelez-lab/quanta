//! Kernel IR type definitions.
//!
//! All core types that represent GPU kernels in platform-agnostic IR form:
//! scalar types, registers, operations, and the top-level [`KernelDef`].

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

/// Memory ordering for `KernelOp::Fence`.
///
/// Mirrors the C11 / Rust `std::sync::atomic::Ordering` semantics. Today
/// only `Fence` consumes this; existing `AtomicOp { .. }` is implicitly
/// SeqCst on every backend. Adding a per-op `order` field to AtomicOp is
/// future work — see `differential_ci.md` (memory) for the rationale.
///
/// Backend mapping (see emit_*/ops.rs):
///   - WGSL has no per-op ordering. Non-Relaxed fences emit
///     `storageBarrier()`; `Relaxed` is a no-op.
///   - MSL: `threadgroup_barrier(mem_flags::mem_device, memory_order_*)`.
///   - SPIR-V: `OpMemoryBarrier` with the appropriate `MemorySemantics`.
///   - CPU: no-op (the interpreter is sequential — every program order is
///     also a memory order).
///   - LLVM: `__atomic_thread_fence` with the matching `__ATOMIC_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
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
    ProtonId {
        dst: Reg,
    },
    NucleusId {
        dst: Reg,
    },
    ProtonSize {
        dst: Reg,
    },

    // Synchronization
    Barrier,
    /// Memory fence with explicit ordering. Applies to the storage class
    /// implied by surrounding atomic ops; backends emit per-spec
    /// equivalents (see `MemoryOrder` doc).
    Fence {
        order: MemoryOrder,
    },
    /// Atomic read-modify-write. The `order` field was added in D-ext.3b.1
    /// to express weaker memory orderings; existing call sites pass
    /// `MemoryOrder::SeqCst` to preserve the prior implicit semantics.
    AtomicOp {
        dst: Reg,
        field: u32,
        index: Reg,
        val: Reg,
        op: AtomicOp,
        ty: ScalarType,
        order: MemoryOrder,
    },
    /// Compare-and-swap. The `order` field was added in D-ext.3b.4
    /// to express per-op memory ordering — same shape as `AtomicOp`
    /// gained in D-ext.3b.1. Existing construction sites pass
    /// `MemoryOrder::SeqCst` to preserve the prior implicit semantics.
    /// LLVM cmpxchg uses the same ordering for both success and
    /// failure paths.
    AtomicCas {
        dst: Reg,
        field: u32,
        index: Reg,
        expected: Reg,
        desired: Reg,
        ty: ScalarType,
        order: MemoryOrder,
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
