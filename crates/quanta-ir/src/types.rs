//! Kernel IR type definitions.
//!
//! All core types that represent GPU kernels in platform-agnostic IR form:
//! scalar types, registers, operations, and the top-level [`KernelDef`].

use crate::quant::QuantScheme;

/// Scalar types supported in GPU kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    F16,
    /// bfloat16: 1 sign / 8 exponent / 7 mantissa bits — the truncated
    /// top 16 bits of an f32, giving f32-range at half width. Native on
    /// backends that support it, otherwise computed in f32 with
    /// round-on-store. Encoding differs from [`F16`] (e5m10); only the
    /// bit-pattern conversions distinguish the two.
    BF16,
    /// fp8 E5M2: 1 sign / 5 exponent (bias 15) / 2 mantissa. The wider-
    /// range 8-bit float (used for gradients). Stored in 8 bits, computed
    /// in f32 (emulated path); pack/unpack rebias + round, unlike bf16's
    /// plain bit-shift.
    FP8E5M2,
    /// fp8 E4M3: 1 sign / 4 exponent (bias 7) / 3 mantissa. The higher-
    /// precision 8-bit float (used for weights/activations).
    FP8E4M3,
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
    /// Signed 4-bit integer. A *logical* 4-bit element in the range
    /// [-8, 7]; physical storage packs 8 nibbles per 32-bit word (the
    /// `store` axis of quantization — see `dtype::int4_{pack,unpack}`).
    /// Computed in i32/f32; used as the storage payload for int4 symmetric
    /// quantization.
    I4,
    Bool,
}

/// Virtual register (SSA-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

/// Role of a cooperative-matrix fragment in a tile GEMM `D = A·B + C`. The role
/// fixes the fragment's logical shape and memory layout when loading/storing:
/// `A` is `m×k` row-major, `B` is `k×n` row-major, `Accumulator` is `m×n`
/// (the `C`/`D` tile). Backends map these to `simdgroup_matrix` /
/// `OpTypeCooperativeMatrixKHR` "use" operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixFrag {
    /// Left operand `A` (`m×k`).
    A,
    /// Right operand `B` (`k×n`).
    B,
    /// Accumulator `C`/`D` (`m×n`).
    Accumulator,
}

/// Constant value.
#[derive(Debug, Clone, Copy)]
pub enum ConstValue {
    F16(u16),
    /// Raw bfloat16 bit pattern (top 16 bits of the f32 encoding).
    BF16(u16),
    /// Raw fp8 E5M2 bit pattern.
    FP8E5M2(u8),
    /// Raw fp8 E4M3 bit pattern.
    FP8E4M3(u8),
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
    /// Rotate left by k bits (k taken mod bit-width). Integer-only.
    Rotl,
    /// Rotate right by k bits (k taken mod bit-width). Integer-only.
    Rotr,
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

/// Whether a math function is a transcendental with no f64 variant on the
/// SPIR-V backend. The GLSL.std.450 transcendentals (Sin/Cos/…/Exp/Log/Pow)
/// accept only 16- or 32-bit floats, so these are refused for f64 rather than
/// emulated (f32 emulation is silently lossy). Sqrt/Rsqrt/Abs/Min/Max/Clamp/
/// Floor/Ceil/Round/Fma accept f64 natively and are not in this set.
pub fn is_f64_transcendental(func: MathFn) -> bool {
    matches!(
        func,
        MathFn::Sin
            | MathFn::Cos
            | MathFn::Tan
            | MathFn::Asin
            | MathFn::Acos
            | MathFn::Atan
            | MathFn::Atan2
            | MathFn::Exp
            | MathFn::Exp2
            | MathFn::Log
            | MathFn::Log2
            | MathFn::Pow
    )
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
    /// Atomic read-modify-write on **workgroup-shared memory** at
    /// `(slot, index)`. Parallels `AtomicOp` for buffer addresses but
    /// targets the shared-memory storage class that backs
    /// `SharedLoad` / `SharedStore`. The `slot` field is a
    /// `SharedDecl` id, NOT a buffer slot.
    ///
    /// Lets kernels build per-bucket histograms in shared memory
    /// without round-tripping through global memory. Each backend
    /// emits its native shared-atomic intrinsic:
    ///   - Metal: `atomic_fetch_add_explicit(&shared[idx], val, …)`
    ///   - SPIR-V: `OpAtomicIAdd` with the `Workgroup` storage class.
    ///   - WGSL: `atomicAdd(&shared[idx], val)`.
    ///   - LLVM: `atomicrmw add ptr addrspace(3)`.
    SharedAtomicOp {
        dst: Reg,
        slot: u32,
        index: Reg,
        val: Reg,
        op: AtomicOp,
        ty: ScalarType,
        order: MemoryOrder,
    },
    /// Compare-and-swap. `success_order` and `failure_order` were
    /// split from a single `order` field as a sustainment item
    /// after D-ext.3b.4. LLVM `cmpxchg` accepts two distinct
    /// orderings (success on the swap, failure on the equality
    /// check) with the constraints `failure ≤ success` and
    /// `failure ∉ {Release, AcqRel}`. Backends that only expose a
    /// single ordering use `success_order` (since success is the
    /// stronger of the two by LLVM's constraint). Existing
    /// construction sites pass the same `MemoryOrder` for both
    /// fields to preserve the prior single-ordering semantics.
    AtomicCas {
        dst: Reg,
        field: u32,
        index: Reg,
        expected: Reg,
        desired: Reg,
        ty: ScalarType,
        success_order: MemoryOrder,
        failure_order: MemoryOrder,
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

    // Quantization (per-tensor symmetric in the first increment). The
    // affine map runs in f32; `scale`/`zero_point` are register operands
    // (loaded from a push-constant for per-tensor; a side buffer later for
    // per-channel). `src`/`dst` of Quantize are f32→int code; Dequantize is
    // int code→f32. zero_point is 0 for Symmetric but carried so Affine is
    // a value change, not a shape change.
    Quantize {
        dst: Reg,
        src: Reg,
        scale: Reg,
        zero_point: Reg,
        scheme: QuantScheme,
    },
    Dequantize {
        dst: Reg,
        src: Reg,
        scale: Reg,
        zero_point: Reg,
        scheme: QuantScheme,
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
    /// Load a cooperative-matrix fragment from a buffer. The fragment is a
    /// subgroup-scoped `m×n×k`-shaped tile (the `frag` role fixes which two of
    /// the three dims apply); `field`/`index` give the source slot and the
    /// element index of the tile's top-left corner, `stride` the row stride (in
    /// elements) of the source matrix. When `from_shared` is set, `field` is a
    /// `SharedDecl` id (threadgroup memory) instead of a buffer slot — this is
    /// how a shared-staged GEMM loads fragments out of the workgroup tile. Each
    /// backend lowers to its native fragment load (`simdgroup_load` /
    /// `OpCooperativeMatrixLoadKHR`).
    CooperativeMatrixLoad {
        dst: Reg,
        field: u32,
        index: Reg,
        stride: Reg,
        frag: MatrixFrag,
        from_shared: bool,
        m: u8,
        n: u8,
        k: u8,
        ty: ScalarType,
    },
    /// Store a cooperative-matrix accumulator fragment to a buffer. Mirrors
    /// `CooperativeMatrixLoad`; `src` is the fragment register, written as an
    /// `m×n` tile at `(field, index)` with row `stride`.
    CooperativeMatrixStore {
        field: u32,
        index: Reg,
        stride: Reg,
        src: Reg,
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
            // bf16 in-body register type is `float`: the emulated path
            // computes in f32 (pack/unpack happens at load/store). The
            // native `bfloat` fork is selected in the emitter when the
            // device advertises it.
            Self::BF16 => "float",
            Self::FP8E5M2 | Self::FP8E4M3 => "float",
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
            // int4 in-body register is `int`: computed as i32, packed into
            // a `uint` word at load/store (8 nibbles per word).
            Self::I4 => "int",
            Self::Bool => "bool",
        }
    }

    /// MSL *buffer storage* element type. Differs from `msl_name` for
    /// bf16 (`ushort`, native 2-byte stride), fp8 (`uchar`, native 1-byte
    /// stride) and int4 (8 nibbles per `uint` word): pack/unpack happens at
    /// load/store. The narrow-float strides match the host upload layout
    /// (tight `Field<u16>` / `Field<u8>`) and the CPU executor; only WGSL
    /// keeps the u32-slot fallback (see `wgsl_storage_name`).
    pub fn msl_storage_name(&self) -> &'static str {
        match self {
            Self::BF16 => "ushort",
            Self::FP8E5M2 | Self::FP8E4M3 => "uchar",
            Self::I4 => "uint",
            _ => self.msl_name(),
        }
    }

    /// WebGPU Shading Language type name.
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            // WGSL has no bf16 type; the in-body register is f32 and
            // bf16 lives only in buffer storage (pack/unpack at I/O).
            Self::BF16 => "f32",
            Self::FP8E5M2 | Self::FP8E4M3 => "f32",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U8 | Self::U16 | Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I8 | Self::I16 | Self::I32 => "i32",
            Self::I64 => "i64",
            // int4 in-body register is `i32`; packed into a `u32` word.
            Self::I4 => "i32",
            Self::Bool => "bool",
        }
    }

    /// WGSL *buffer storage* element type. Differs from `wgsl_name` for
    /// bf16/fp8 (one per `u32` word) and int4 (8 nibbles per `u32` word):
    /// pack/unpack at load/store.
    ///
    /// WGSL storage buffers cannot hold 16-/8-bit array elements, so bf16
    /// and fp8 keep the u32-slot layout here even though MSL/SPIR-V (and
    /// the host/CPU) use native stride. Hosts feeding the WebGPU backend
    /// must expand tight narrow data one-element-per-word (see
    /// `Gpu::narrow_storage_u32_slot`).
    pub fn wgsl_storage_name(&self) -> &'static str {
        match self {
            Self::BF16 => "u32",
            Self::FP8E5M2 | Self::FP8E4M3 => "u32",
            Self::I4 => "u32",
            _ => self.wgsl_name(),
        }
    }
}
