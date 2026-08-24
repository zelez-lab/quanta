//! Kernel IR type definitions.
//!
//! All core types that represent GPU kernels in platform-agnostic IR form:
//! scalar types, registers, operations, and the top-level [`KernelDef`].

use crate::quant::QuantScheme;

/// Scalar types supported in GPU kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// IEEE 754 binary16 (e5m10) — 1 sign / 5 exponent / 10 mantissa bits.
    F16,
    /// bfloat16: 1 sign / 8 exponent / 7 mantissa bits — the truncated
    /// top 16 bits of an f32, giving f32-range at half width. Native on
    /// backends that support it, otherwise computed in f32 with
    /// round-on-store. Encoding differs from [`ScalarType::F16`] (e5m10); only the
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
    /// IEEE 754 binary32 — the default float on every backend.
    F32,
    /// IEEE 754 binary64. The transcendentals listed by
    /// [`is_f64_transcendental`] have no SPIR-V form at this width.
    F64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer — the natural GPU word.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Signed 4-bit integer. A *logical* 4-bit element in the range
    /// [-8, 7]; physical storage packs 8 nibbles per 32-bit word (the
    /// `store` axis of quantization — see `dtype::int4_{pack,unpack}`).
    /// Computed in i32/f32; used as the storage payload for int4 symmetric
    /// quantization.
    I4,
    /// Boolean. A register-only type — buffers carry it as an integer.
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
    /// Raw half-precision bit pattern.
    F16(u16),
    /// Raw bfloat16 bit pattern (top 16 bits of the f32 encoding).
    BF16(u16),
    /// Raw fp8 E5M2 bit pattern.
    FP8E5M2(u8),
    /// Raw fp8 E4M3 bit pattern.
    FP8E4M3(u8),
    /// Single-precision literal.
    F32(f32),
    /// Double-precision literal.
    F64(f64),
    /// Unsigned literal — also carries the narrower unsigned widths.
    U32(u32),
    /// Unsigned 64-bit literal.
    U64(u64),
    /// Signed literal — also carries the narrower signed widths.
    I32(i32),
    /// Signed 64-bit literal.
    I64(i64),
    /// Boolean literal.
    Bool(bool),
}

/// Kernel parameter — how function arguments map to GPU bindings.
#[derive(Debug, Clone)]
pub enum KernelParam {
    /// `&[T]` — a read-only storage buffer.
    FieldRead {
        /// Parameter name, reused as the binding's name in emitted source.
        name: String,
        /// Binding slot: the positional parameter index, shared across the
        /// buffer / texture / constant namespace.
        slot: u32,
        /// Element type of the buffer.
        scalar_type: ScalarType,
    },
    /// `&mut [T]` — a read-write storage buffer. These are the slots
    /// [`field_write_mask`] reports.
    FieldWrite {
        /// Parameter name, reused as the binding's name in emitted source.
        name: String,
        /// Binding slot: the positional parameter index, shared across the
        /// buffer / texture / constant namespace.
        slot: u32,
        /// Element type of the buffer.
        scalar_type: ScalarType,
    },
    /// A scalar passed by value, delivered as a push constant / uniform.
    Constant {
        /// Parameter name, reused as the binding's name in emitted source.
        name: String,
        /// Binding slot: the positional parameter index, shared across the
        /// buffer / texture / constant namespace.
        slot: u32,
        /// Type of the scalar.
        scalar_type: ScalarType,
    },
    /// `&Texture2D<T>` — read-only texel access. A storage image whose
    /// variable is decorated NonWritable: reads are `texture_load_2d`
    /// (OpImageRead / MSL `access::read` / WGSL `read`), writes are
    /// rejected. Same scalar-driven format contract as the read-write
    /// form; unlike it, packed-RGBA8 reads need no Metal read-write
    /// texture tier.
    Texture2DRead {
        /// Parameter name, reused as the binding's name in emitted source.
        name: String,
        /// Binding slot: the positional parameter index, shared across the
        /// buffer / texture / constant namespace.
        slot: u32,
        /// Texel scalar, which fixes the storage format — see
        /// [`ScalarType::spirv_storage_image_format`].
        scalar_type: ScalarType,
    },
    /// `&mut Texture2D<T>` — read-write texel access on a storage image.
    Texture2DReadWrite {
        /// Parameter name, reused as the binding's name in emitted source.
        name: String,
        /// Binding slot: the positional parameter index, shared across the
        /// buffer / texture / constant namespace.
        slot: u32,
        /// Texel scalar, which fixes the storage format — see
        /// [`ScalarType::spirv_storage_image_format`].
        scalar_type: ScalarType,
    },
    /// `&Sampled2D<T>` — sampled access through the fixed
    /// NEAREST/CLAMP_TO_EDGE sampler (combined image+sampler binding).
    Sampled2D {
        /// Parameter name, reused as the binding's name in emitted source.
        name: String,
        /// Binding slot: the positional parameter index, shared across the
        /// buffer / texture / constant namespace.
        slot: u32,
        /// Sampled type. Only `f32` is wired — see
        /// [`reject_sampled_u32_texture`].
        scalar_type: ScalarType,
    },
    /// `&Sampled3D<T>` — the 3D sampled form. There is no 3D texel form
    /// yet; `&Texture3D` is a parse error until one is wired.
    Sampled3D {
        /// Parameter name, reused as the binding's name in emitted source.
        name: String,
        /// Binding slot: the positional parameter index, shared across the
        /// buffer / texture / constant namespace.
        slot: u32,
        /// Sampled type.
        scalar_type: ScalarType,
    },
}

/// One cooperative multiply-accumulate a kernel performs, with the element
/// type of each operand resolved through the fragment registers that feed
/// it: `D[m×n] = A[m×k] · B[k×n] + C[m×n]`. This is the unit a device
/// shape (`Gpu::cooperative_matrix_shapes`) has to match exactly — the
/// hardware forms are mixed (f16/bf16 inputs, f32 accumulation), so the
/// per-op `ty` fields are not a shape by themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoopMmaUse {
    /// Rows of `A` and of the accumulator.
    pub m: u8,
    /// Columns of `B` and of the accumulator.
    pub n: u8,
    /// Shared inner dimension — columns of `A`, rows of `B`.
    pub k: u8,
    /// Element type of the `A` fragment, resolved through the register that
    /// wrote it.
    pub a_ty: ScalarType,
    /// Element type of the `B` fragment, resolved through the register that
    /// wrote it.
    pub b_ty: ScalarType,
    /// Element type of the `C` accumulator, resolved through the register
    /// that wrote it.
    pub c_ty: ScalarType,
    /// Element type of the `D` result the op writes.
    pub result_ty: ScalarType,
}

/// One fragment a kernel loads or stores without (necessarily) feeding an
/// MMA: its role, shape and element type. A device must list some shape
/// with that `m,n,k` whose corresponding slot (`A`/`B` → inputs,
/// `Accumulator` → `C` or `D`) has that type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoopFragUse {
    /// Which operand slot of `D = A·B + C` the fragment fills.
    pub frag: MatrixFrag,
    /// Rows of `A` and of the accumulator.
    pub m: u8,
    /// Columns of `B` and of the accumulator.
    pub n: u8,
    /// Shared inner dimension — columns of `A`, rows of `B`.
    pub k: u8,
    /// Element type of the fragment.
    pub ty: ScalarType,
}

/// Everything a kernel asks of a device's cooperative-matrix support.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoopMatrixUses {
    /// Every distinct multiply-accumulate shape the kernel performs.
    pub mmas: Vec<CoopMmaUse>,
    /// Every distinct fragment shape the kernel loads or stores, including
    /// those that feed no MMA.
    pub frags: Vec<CoopFragUse>,
}

impl CoopMatrixUses {
    /// Whether the kernel asks nothing of cooperative-matrix support, in
    /// which case a driver can skip the shape match entirely.
    pub fn is_empty(&self) -> bool {
        self.mmas.is_empty() && self.frags.is_empty()
    }
}

/// Collect a kernel's cooperative-matrix uses — body, nested arms, device
/// functions — resolving each MMA operand's element type through the
/// register it reads (written by a fragment load or an earlier MMA).
/// Drivers match the result against the device's enumerated shapes before
/// creating a pipeline, so a kernel built for a shape the hardware lacks
/// is refused with a clear `NotSupported` instead of failing (or silently
/// misexecuting) inside the driver. An operand whose register no
/// cooperative op wrote (malformed IR) is reported as the MMA's own `ty`,
/// which the shape match then rejects honestly.
pub fn cooperative_matrix_uses(def: &KernelDef) -> CoopMatrixUses {
    use std::collections::HashMap;
    fn walk(ops: &[KernelOp], reg_ty: &mut HashMap<u32, ScalarType>, out: &mut CoopMatrixUses) {
        for op in ops {
            match op {
                KernelOp::CooperativeMatrixLoad {
                    dst,
                    frag,
                    m,
                    n,
                    k,
                    ty,
                    ..
                } => {
                    reg_ty.insert(dst.0, *ty);
                    push_unique(
                        &mut out.frags,
                        CoopFragUse {
                            frag: *frag,
                            m: *m,
                            n: *n,
                            k: *k,
                            ty: *ty,
                        },
                    );
                }
                KernelOp::CooperativeMatrixStore { m, n, k, ty, .. } => {
                    push_unique(
                        &mut out.frags,
                        CoopFragUse {
                            frag: MatrixFrag::Accumulator,
                            m: *m,
                            n: *n,
                            k: *k,
                            ty: *ty,
                        },
                    );
                }
                KernelOp::CooperativeMMA {
                    dst,
                    a,
                    b,
                    c,
                    m,
                    n,
                    k,
                    ty,
                } => {
                    let of = |r: &Reg| reg_ty.get(&r.0).copied().unwrap_or(*ty);
                    push_unique(
                        &mut out.mmas,
                        CoopMmaUse {
                            m: *m,
                            n: *n,
                            k: *k,
                            a_ty: of(a),
                            b_ty: of(b),
                            c_ty: of(c),
                            result_ty: *ty,
                        },
                    );
                    reg_ty.insert(dst.0, *ty);
                }
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    walk(then_ops, reg_ty, out);
                    walk(else_ops, reg_ty, out);
                }
                KernelOp::Loop { body, .. } => walk(body, reg_ty, out),
                _ => {}
            }
        }
    }
    fn push_unique<T: PartialEq>(v: &mut Vec<T>, x: T) {
        if !v.contains(&x) {
            v.push(x);
        }
    }
    let mut out = CoopMatrixUses::default();
    let mut reg_ty = HashMap::new();
    walk(&def.body, &mut reg_ty, &mut out);
    for f in &def.device_functions {
        let mut fn_regs = HashMap::new();
        walk(&f.body, &mut fn_regs, &mut out);
    }
    out
}

/// Bit N set = binding slot N is a [`KernelParam::FieldWrite`] — the
/// kernel may WRITE (and read: `&mut [T]` is read-write) that buffer.
/// Clear bits with a bound field are read-only. The deferred lane uses
/// this to order only genuinely dependent dispatches; drivers stamp it
/// onto the `Wave` at JIT time, the `#[quanta::kernel]` wrapper stamps
/// it from the signature.
pub fn field_write_mask(def: &KernelDef) -> u16 {
    let mut mask = 0u16;
    for p in &def.params {
        if let KernelParam::FieldWrite { slot, .. } = p
            && *slot < 16
        {
            mask |= 1 << *slot;
        }
    }
    mask
}

/// gpu_print record-buffer geometry, shared by every emitter and
/// driver implementing the scheme. Word 0 is the atomic cursor; a
/// print appends a 3-word record (quark, type tag, raw bits) at
/// cursor+1, dropped when its last word would land at or past this
/// cap. Metal's guarded store allocates exactly this many words; the
/// SPIR-V lowering is branchless (OpSelect redirects an overflowing
/// record to a dead tail at the cap), so Vulkan allocates
/// `DEBUG_PRINT_CAP_WORDS + 4`.
pub const DEBUG_PRINT_CAP_WORDS: u32 = 16384;

/// The reserved slot for the record buffer: Metal buffer index and
/// Vulkan descriptor binding (set 0). User fields stop at slot 15,
/// so the reservation can never collide.
pub const DEBUG_PRINT_BINDING: u32 = 30;

/// Whether the body contains a `DebugPrint`, at any nesting depth
/// (Branch arms and Loop bodies included) — the drivers, the macro
/// and the emitters all key the debug-buffer machinery on this.
pub fn body_contains_debug_print(ops: &[KernelOp]) -> bool {
    ops.iter().any(|op| match op {
        KernelOp::DebugPrint { .. } => true,
        KernelOp::Branch {
            then_ops, else_ops, ..
        } => body_contains_debug_print(then_ops) || body_contains_debug_print(else_ops),
        KernelOp::Loop { body, .. } => body_contains_debug_print(body),
        _ => false,
    })
}

/// Binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / b`. Integer division by zero yields 0 on every backend — the
    /// emitters substitute a safe divisor rather than trusting hardware.
    Div,
    /// `a % b`. Integer remainder by zero yields 0, by the same guard as
    /// [`BinOp::Div`].
    Rem,
    /// Bitwise `a & b`. Integer-only.
    BitAnd,
    /// Bitwise `a | b`. Integer-only.
    BitOr,
    /// Bitwise `a ^ b`. Integer-only.
    BitXor,
    /// Left shift `a << b`. Integer-only.
    Shl,
    /// Right shift `a >> b` — arithmetic for signed types, logical for
    /// unsigned. Integer-only.
    Shr,
    /// Rotate left by k bits (k taken mod bit-width). Integer-only.
    Rotl,
    /// Rotate right by k bits (k taken mod bit-width). Integer-only.
    Rotr,
    /// Addition that clamps at the type's bounds instead of wrapping.
    SatAdd,
    /// Subtraction that clamps at the type's bounds instead of wrapping.
    SatSub,
}

/// Unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Arithmetic negation `-a`.
    Neg,
    /// Bitwise complement `!a`. Integer-only.
    BitNot,
    /// Boolean negation `!a`. Operates on a [`ScalarType::Bool`] register.
    LogicalNot,
}

/// Comparison operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `a == b`.
    Eq,
    /// `a != b`.
    Ne,
    /// `a < b`.
    Lt,
    /// `a <= b`.
    Le,
    /// `a > b`.
    Gt,
    /// `a >= b`.
    Ge,
}

/// Atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    /// Fetch-and-add; the destination register receives the old value.
    Add,
    /// Fetch-and-subtract; the destination register receives the old value.
    Sub,
    /// Store the smaller of the current and given value, returning the old.
    Min,
    /// Store the larger of the current and given value, returning the old.
    Max,
    /// Fetch-and-`and`, returning the old value.
    And,
    /// Fetch-and-`or`, returning the old value.
    Or,
    /// Fetch-and-`xor`, returning the old value.
    Xor,
    /// Unconditional swap, returning the old value.
    Exchange,
    /// Compare-and-swap. Reached through [`KernelOp::AtomicCas`], which
    /// carries the extra expected/desired operands and the two orderings.
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
    /// No ordering beyond the operation's own atomicity.
    Relaxed,
    /// Later reads and writes cannot be hoisted above this one.
    Acquire,
    /// Earlier reads and writes cannot sink below this one.
    Release,
    /// Both [`MemoryOrder::Acquire`] and [`MemoryOrder::Release`].
    AcqRel,
    /// Sequential consistency — a single total order over all such
    /// operations. The implicit ordering of every pre-existing atomic.
    SeqCst,
}

/// Built-in math functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathFn {
    /// Sine, argument in radians.
    Sin,
    /// Cosine, argument in radians.
    Cos,
    /// Tangent, argument in radians.
    Tan,
    /// Arcsine, result in radians.
    Asin,
    /// Arccosine, result in radians.
    Acos,
    /// Arctangent, result in radians.
    Atan,
    /// Two-argument arctangent `atan2(y, x)`, quadrant-correct.
    Atan2,
    /// Square root.
    Sqrt,
    /// Reciprocal square root `1/sqrt(x)`, the hardware's fast form.
    Rsqrt,
    /// `e^x`.
    Exp,
    /// `2^x`.
    Exp2,
    /// Natural logarithm.
    Log,
    /// Base-2 logarithm.
    Log2,
    /// `pow(x, y)`.
    Pow,
    /// Absolute value.
    Abs,
    /// Two-argument minimum.
    Min,
    /// Two-argument maximum.
    Max,
    /// Three-argument `clamp(x, lo, hi)`.
    Clamp,
    /// Round towards negative infinity.
    Floor,
    /// Round towards positive infinity.
    Ceil,
    /// Round to the nearest integer.
    Round,
    /// Fused multiply-add `a * b + c`, rounded once.
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
    /// Read one element out of a bound buffer.
    Load {
        /// Register receiving the loaded element.
        dst: Reg,
        /// Binding slot of the buffer.
        field: u32,
        /// Element index into the buffer.
        index: Reg,
        /// Element type — narrow types unpack from their storage word here.
        ty: ScalarType,
    },
    /// Write one element into a bound `&mut [T]` buffer.
    Store {
        /// Binding slot of the buffer.
        field: u32,
        /// Element index into the buffer.
        index: Reg,
        /// Register holding the value to write.
        src: Reg,
        /// Element type — narrow types pack into their storage word here.
        ty: ScalarType,
    },
    /// Declare a fixed-size workgroup-shared array.
    SharedDecl {
        /// Shared-array id, the handle the shared load/store ops address.
        id: u32,
        /// Element type of the array.
        ty: ScalarType,
        /// Number of elements, fixed at compile time.
        count: u32,
    },
    /// Read one element out of a shared array.
    SharedLoad {
        /// Register receiving the loaded element.
        dst: Reg,
        /// Shared-array id from a `SharedDecl` / `SharedDeclDyn`.
        id: u32,
        /// Element index into the array.
        index: Reg,
        /// Element type.
        ty: ScalarType,
    },
    /// Write one element into a shared array.
    SharedStore {
        /// Shared-array id from a `SharedDecl` / `SharedDeclDyn`.
        id: u32,
        /// Element index into the array.
        index: Reg,
        /// Register holding the value to write.
        src: Reg,
        /// Element type.
        ty: ScalarType,
    },

    // Arithmetic
    /// Two-operand arithmetic or bitwise operation.
    BinOp {
        /// Register receiving the result.
        dst: Reg,
        /// Left operand.
        a: Reg,
        /// Right operand.
        b: Reg,
        /// Which operation to perform.
        op: BinOp,
        /// Type both operands and the result share.
        ty: ScalarType,
    },
    /// One-operand arithmetic or bitwise operation.
    UnaryOp {
        /// Register receiving the result.
        dst: Reg,
        /// The operand.
        a: Reg,
        /// Which operation to perform.
        op: UnaryOp,
        /// Type the operand and the result share.
        ty: ScalarType,
    },
    /// Compare two registers, producing a [`ScalarType::Bool`] result.
    Cmp {
        /// Register receiving the boolean result.
        dst: Reg,
        /// Left operand.
        a: Reg,
        /// Right operand.
        b: Reg,
        /// Which comparison to perform.
        op: CmpOp,
        /// Type of the two operands — the result is always `Bool`.
        ty: ScalarType,
    },

    // Control flow
    /// Structured `if`/`else`. Both arms are nested op lists, so the IR
    /// stays a tree rather than a basic-block graph.
    Branch {
        /// Boolean register selecting the arm.
        cond: Reg,
        /// Ops run when `cond` is true.
        then_ops: Vec<KernelOp>,
        /// Ops run when `cond` is false; empty for a bare `if`.
        else_ops: Vec<KernelOp>,
    },
    /// Counted loop running `count` iterations, exitable with `Break`.
    Loop {
        /// Register holding the trip count.
        count: Reg,
        /// Register the loop writes with the current iteration index.
        iter_reg: Reg,
        /// Ops making up the loop body.
        body: Vec<KernelOp>,
    },

    // Math
    /// Call a built-in math function.
    MathCall {
        /// Register receiving the result.
        dst: Reg,
        /// Which built-in to call.
        func: MathFn,
        /// Argument registers, in the function's own order.
        args: Vec<Reg>,
        /// Type the arguments and result share.
        ty: ScalarType,
    },

    // Thread indexing
    /// This thread's global index across the whole dispatch.
    QuarkId {
        /// Register receiving the index.
        dst: Reg,
    },
    /// Total number of threads in the dispatch.
    QuarkCount {
        /// Register receiving the count.
        dst: Reg,
    },
    /// This thread's index within its workgroup.
    ProtonId {
        /// Register receiving the index.
        dst: Reg,
    },
    /// This workgroup's index within the dispatch.
    NucleusId {
        /// Register receiving the index.
        dst: Reg,
    },
    /// Number of threads per workgroup — the kernel's `workgroup_size`
    /// product.
    ProtonSize {
        /// Register receiving the size.
        dst: Reg,
    },

    // Synchronization
    /// Workgroup execution + memory barrier: every thread in the workgroup
    /// waits here, and shared writes made before it are visible after.
    Barrier,
    /// Memory fence with explicit ordering. Applies to the storage class
    /// implied by surrounding atomic ops; backends emit per-spec
    /// equivalents (see `MemoryOrder` doc).
    Fence {
        /// Ordering the fence establishes.
        order: MemoryOrder,
    },
    /// Atomic read-modify-write. The `order` field was added in D-ext.3b.1
    /// to express weaker memory orderings; existing call sites pass
    /// `MemoryOrder::SeqCst` to preserve the prior implicit semantics.
    AtomicOp {
        /// Register receiving the value held before the update.
        dst: Reg,
        /// Binding slot of the target buffer.
        field: u32,
        /// Element index into the buffer.
        index: Reg,
        /// Register holding the operand applied to the current value.
        val: Reg,
        /// Which read-modify-write to perform.
        op: AtomicOp,
        /// Element type.
        ty: ScalarType,
        /// Memory ordering of the update.
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
        /// Register receiving the value held before the update.
        dst: Reg,
        /// Shared-array id from a `SharedDecl`, NOT a buffer slot.
        slot: u32,
        /// Element index into the shared array.
        index: Reg,
        /// Register holding the operand applied to the current value.
        val: Reg,
        /// Which read-modify-write to perform.
        op: AtomicOp,
        /// Element type.
        ty: ScalarType,
        /// Memory ordering of the update.
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
        /// Register receiving the value held before the attempt, which the
        /// caller compares against `expected` to learn whether it swapped.
        dst: Reg,
        /// Binding slot of the target buffer.
        field: u32,
        /// Element index into the buffer.
        index: Reg,
        /// Value the swap is conditional on.
        expected: Reg,
        /// Value written when the comparison succeeds.
        desired: Reg,
        /// Element type.
        ty: ScalarType,
        /// Ordering applied when the swap happens.
        success_order: MemoryOrder,
        /// Ordering applied when the comparison fails — must be no stronger
        /// than `success_order`, and neither `Release` nor `AcqRel`.
        failure_order: MemoryOrder,
    },

    // Warp/wave
    /// Butterfly exchange within the subgroup: this lane reads the value
    /// held by lane `self ^ lane_delta`.
    WaveShuffle {
        /// Register receiving the partner lane's value.
        dst: Reg,
        /// Register whose value this lane contributes.
        src: Reg,
        /// XOR mask picking the partner lane.
        lane_delta: Reg,
        /// Type of the exchanged value.
        ty: ScalarType,
    },
    /// Bitmask of the subgroup lanes whose predicate is true, low bit =
    /// lane 0. Truncated to 32 bits.
    WaveBallot {
        /// Register receiving the mask.
        dst: Reg,
        /// Per-lane predicate.
        predicate: Reg,
    },
    /// Whether the predicate holds in at least one lane of the subgroup.
    WaveAny {
        /// Register receiving the boolean result.
        dst: Reg,
        /// Per-lane predicate.
        predicate: Reg,
    },
    /// Whether the predicate holds in every lane of the subgroup.
    WaveAll {
        /// Register receiving the boolean result.
        dst: Reg,
        /// Per-lane predicate.
        predicate: Reg,
    },

    // Type conversion
    /// Value-preserving conversion between scalar types — `as` in the DSL.
    Cast {
        /// Register receiving the converted value.
        dst: Reg,
        /// Register holding the value to convert.
        src: Reg,
        /// Type the source register holds.
        from: ScalarType,
        /// Type to convert to.
        to: ScalarType,
    },
    /// Materialize a literal into a register.
    Const {
        /// Register receiving the literal.
        dst: Reg,
        /// The literal, which also fixes the register's type.
        value: ConstValue,
    },

    // Quantization (per-tensor symmetric in the first increment). The
    // affine map runs in f32; `scale`/`zero_point` are register operands
    // (loaded from a push-constant for per-tensor; a side buffer later for
    // per-channel). `src`/`dst` of Quantize are f32→int code; Dequantize is
    // int code→f32. zero_point is 0 for Symmetric but carried so Affine is
    // a value change, not a shape change.
    /// Map an f32 value to its integer code under `scheme`.
    Quantize {
        /// Register receiving the integer code.
        dst: Reg,
        /// Register holding the f32 value.
        src: Reg,
        /// Register holding the affine scale.
        scale: Reg,
        /// Register holding the affine zero point — 0 under a symmetric
        /// scheme, but carried so that affine is a value change only.
        zero_point: Reg,
        /// Level and mode the mapping follows.
        scheme: QuantScheme,
    },
    /// Map an integer code back to f32 under `scheme`.
    Dequantize {
        /// Register receiving the f32 value.
        dst: Reg,
        /// Register holding the integer code.
        src: Reg,
        /// Register holding the affine scale.
        scale: Reg,
        /// Register holding the affine zero point — 0 under a symmetric
        /// scheme, but carried so that affine is a value change only.
        zero_point: Reg,
        /// Level and mode the mapping follows.
        scheme: QuantScheme,
    },

    // Vector
    /// Pack scalar registers into a vector register; the component count
    /// sets the vector's width.
    VecConstruct {
        /// Register receiving the vector.
        dst: Reg,
        /// Component registers, in order.
        components: Vec<Reg>,
        /// Element type of every component.
        ty: ScalarType,
    },
    /// Read one component out of a vector register.
    VecExtract {
        /// Register receiving the component.
        dst: Reg,
        /// Register holding the vector.
        vec: Reg,
        /// Zero-based component index.
        component: u8,
        /// Element type of the vector.
        ty: ScalarType,
    },
    /// Multiply two matrix-typed registers.
    MatMul {
        /// Register receiving the product.
        dst: Reg,
        /// Left operand.
        a: Reg,
        /// Right operand.
        b: Reg,
        /// Square dimension of the operands. Carried for completeness — no
        /// emitter reads it, since the operand registers already carry their
        /// own matrix type.
        size: u8,
        /// Element type.
        ty: ScalarType,
    },
    /// Cooperative matrix multiply-accumulate (tensor cores / SIMD group matrix).
    /// D = A * B + C where A, B, C, D are SIMD-group-scoped matrices.
    CooperativeMMA {
        /// Fragment register receiving `D`.
        dst: Reg,
        /// Fragment register holding `A`.
        a: Reg,
        /// Fragment register holding `B`.
        b: Reg,
        /// Fragment register holding the `C` accumulator.
        c: Reg,
        /// Rows of `A` and of the accumulator.
        m: u8,
        /// Columns of `B` and of the accumulator.
        n: u8,
        /// Shared inner dimension — columns of `A`, rows of `B`.
        k: u8,
        /// Element type of `D`. Operand types are resolved through the
        /// registers that wrote them, since hardware forms are mixed.
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
        /// Fragment register receiving the tile.
        dst: Reg,
        /// Buffer binding slot, or a `SharedDecl` id when `from_shared`.
        field: u32,
        /// Element index of the tile's top-left corner.
        index: Reg,
        /// Row stride of the source matrix, in elements.
        stride: Reg,
        /// Which operand slot of `D = A·B + C` the tile fills.
        frag: MatrixFrag,
        /// Whether `field` names threadgroup memory rather than a buffer.
        from_shared: bool,
        /// Rows of `A` and of the accumulator.
        m: u8,
        /// Columns of `B` and of the accumulator.
        n: u8,
        /// Shared inner dimension — columns of `A`, rows of `B`.
        k: u8,
        /// Element type of the fragment.
        ty: ScalarType,
    },
    /// Store a cooperative-matrix accumulator fragment to a buffer. Mirrors
    /// `CooperativeMatrixLoad`; `src` is the fragment register, written as an
    /// `m×n` tile at `(field, index)` with row `stride`.
    CooperativeMatrixStore {
        /// Buffer binding slot to write.
        field: u32,
        /// Element index of the tile's top-left corner.
        index: Reg,
        /// Row stride of the destination matrix, in elements.
        stride: Reg,
        /// Fragment register holding the accumulator tile.
        src: Reg,
        /// Rows of the accumulator.
        m: u8,
        /// Columns of the accumulator.
        n: u8,
        /// Shared inner dimension of the GEMM the tile came from.
        k: u8,
        /// Element type of the fragment.
        ty: ScalarType,
    },

    // Texture
    /// Sample a `&Sampled2D` through the fixed NEAREST/CLAMP_TO_EDGE
    /// sampler. Refused against a texel slot — see
    /// [`reject_sample_on_storage`].
    TextureSample2D {
        /// Register receiving the sampled texel.
        dst: Reg,
        /// Binding slot of the texture.
        texture: u32,
        /// Horizontal coordinate.
        x: Reg,
        /// Vertical coordinate.
        y: Reg,
        /// Type of the sampled value.
        ty: ScalarType,
    },
    /// Sample a `&Sampled3D` through the fixed NEAREST/CLAMP_TO_EDGE
    /// sampler.
    TextureSample3D {
        /// Register receiving the sampled texel.
        dst: Reg,
        /// Binding slot of the texture.
        texture: u32,
        /// Horizontal coordinate.
        x: Reg,
        /// Vertical coordinate.
        y: Reg,
        /// Depth coordinate.
        z: Reg,
        /// Type of the sampled value.
        ty: ScalarType,
    },
    /// Write one texel of a `&mut Texture2D`. Refused against a read-only
    /// texel slot — see [`reject_write_on_read_only`].
    TextureWrite2D {
        /// Binding slot of the texture.
        texture: u32,
        /// Horizontal coordinate.
        x: Reg,
        /// Vertical coordinate.
        y: Reg,
        /// Register holding the texel to write.
        value: Reg,
        /// Texel type, which fixes the storage format.
        ty: ScalarType,
    },
    /// Query a texture's dimensions.
    TextureSize {
        /// Register receiving the width in texels.
        dst_w: Reg,
        /// Register receiving the height in texels.
        dst_h: Reg,
        /// Binding slot of the texture.
        texture: u32,
    },

    // Register copy (for loop-carried variable updates)
    /// Move a value between registers. How a loop-carried variable is
    /// updated: the destination is a mutable cell, not a fresh SSA name.
    Copy {
        /// Register written.
        dst: Reg,
        /// Register read.
        src: Reg,
        /// Type of the value.
        ty: ScalarType,
    },

    // Control flow
    /// Leave the innermost enclosing `Loop`.
    Break,

    // Dynamic parallelism
    /// Launch a nested dispatch from inside the kernel. No backend
    /// implements it: the shader emitters drop it and the LLVM path
    /// refuses it, so the host must enqueue the child wave instead.
    Dispatch {
        /// Register holding the wave handle to launch.
        wave: Reg,
        /// Workgroup counts along x, y and z.
        groups: [Reg; 3],
    },

    // Device function call (user-defined helper)
    /// Call a `#[quanta::device]` helper defined alongside the kernel.
    DeviceCall {
        /// Register receiving the return value.
        dst: Reg,
        /// Name of the [`DeviceFnDef`] to call.
        func_name: String,
        /// Argument registers, in declaration order.
        args: Vec<Reg>,
        /// Return type of the helper.
        ty: ScalarType,
    },

    // Bit manipulation
    /// Reinterpret a register's bits at another type of the same width.
    Bitcast {
        /// Register receiving the reinterpreted value.
        dst: Reg,
        /// Register holding the bits.
        src: Reg,
        /// Type the source register holds.
        from: ScalarType,
        /// Type to reinterpret as.
        to: ScalarType,
    },
    /// Number of zero bits below the lowest set bit.
    CountTrailingZeros {
        /// Register receiving the count.
        dst: Reg,
        /// Register holding the value.
        src: Reg,
        /// Integer type, which fixes the bit width.
        ty: ScalarType,
    },
    /// Number of zero bits above the highest set bit.
    CountLeadingZeros {
        /// Register receiving the count.
        dst: Reg,
        /// Register holding the value.
        src: Reg,
        /// Integer type, which fixes the bit width.
        ty: ScalarType,
    },
    /// Number of set bits.
    PopCount {
        /// Register receiving the count.
        dst: Reg,
        /// Register holding the value.
        src: Reg,
        /// Integer type, which fixes the bit width.
        ty: ScalarType,
    },

    // Dot product (vector)
    /// Dot product of two vector registers.
    Dot {
        /// Register receiving the scalar result.
        dst: Reg,
        /// Left operand.
        a: Reg,
        /// Right operand.
        b: Reg,
        /// Element type.
        ty: ScalarType,
        /// Vector width. Carried for completeness — no emitter reads it,
        /// since the operand registers already carry their own vector type.
        width: u8,
    },

    // Subgroup
    /// Number of lanes in a subgroup on this device.
    SubgroupSize {
        /// Register receiving the size.
        dst: Reg,
    },
    /// This thread's lane index within its subgroup, `0..SubgroupSize`.
    /// A real builtin on every backend (`SubgroupLocalInvocationId`,
    /// `thread_index_in_simdgroup`, `subgroup_invocation_id`); on the CPU
    /// executor it is `proton_id % SUBGROUP_SIZE`, the grouping its warp
    /// cohorts use. Never `ProtonId` — the two agree only while the
    /// workgroup fits in one subgroup.
    SubgroupLaneId {
        /// Register receiving the lane index.
        dst: Reg,
    },

    // Subgroup scan/reduce
    /// Sum of the contributed values across the subgroup, broadcast to
    /// every lane.
    SubgroupReduceAdd {
        /// Register receiving the reduction.
        dst: Reg,
        /// Register whose value this lane contributes.
        src: Reg,
        /// Type of the values.
        ty: ScalarType,
    },
    /// Minimum across the subgroup, broadcast to every lane.
    SubgroupReduceMin {
        /// Register receiving the reduction.
        dst: Reg,
        /// Register whose value this lane contributes.
        src: Reg,
        /// Type of the values.
        ty: ScalarType,
    },
    /// Maximum across the subgroup, broadcast to every lane.
    SubgroupReduceMax {
        /// Register receiving the reduction.
        dst: Reg,
        /// Register whose value this lane contributes.
        src: Reg,
        /// Type of the values.
        ty: ScalarType,
    },
    /// Prefix sum over the subgroup, excluding this lane's own value —
    /// lane 0 receives the identity.
    SubgroupExclusiveAdd {
        /// Register receiving the prefix sum.
        dst: Reg,
        /// Register whose value this lane contributes.
        src: Reg,
        /// Type of the values.
        ty: ScalarType,
    },
    /// Prefix sum over the subgroup, including this lane's own value.
    SubgroupInclusiveAdd {
        /// Register receiving the prefix sum.
        dst: Reg,
        /// Register whose value this lane contributes.
        src: Reg,
        /// Type of the values.
        ty: ScalarType,
    },

    // Texture load without sampler
    /// Read one texel of a `&Texture2D` / `&mut Texture2D` by integer
    /// coordinate. The only way to read a texel slot — sampling one is
    /// refused.
    TextureLoad2D {
        /// Register receiving the texel.
        dst: Reg,
        /// Binding slot of the texture.
        texture: u32,
        /// Horizontal coordinate.
        x: Reg,
        /// Vertical coordinate.
        y: Reg,
        /// Texel type, which fixes the storage format.
        ty: ScalarType,
    },

    // Dynamic shared memory declaration (size determined at dispatch)
    /// Declare a shared array whose length comes from the dispatch's
    /// `dynamic_shared_bytes` rather than from the kernel source.
    SharedDeclDyn {
        /// Shared-array id, the handle the shared load/store ops address.
        id: u32,
        /// Element type of the array.
        ty: ScalarType,
    },

    // GPU debug print (writes value + thread_id to a debug buffer)
    /// Append `(quark_id, value)` to the kernel's debug buffer, for
    /// inspecting a running kernel from the host.
    DebugPrint {
        /// Register holding the value to record.
        src: Reg,
        /// Type of the value, which fixes how it is widened to a word.
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
    /// Function name, as `KernelOp::DeviceCall` spells it.
    pub name: String,
    /// Parameters as `(name, type)` pairs, in declaration order.
    pub params: Vec<(String, ScalarType)>,
    /// Return type.
    pub return_type: ScalarType,
    /// The function body.
    pub body: Vec<KernelOp>,
    /// Next unused register number in the function's own register space.
    pub next_reg: u32,
}

/// Complete kernel definition in IR form.
#[derive(Debug, Clone)]
pub struct KernelDef {
    /// Kernel name, which becomes the entry point's name in emitted source.
    pub name: String,
    /// Bindings the kernel takes, in positional order.
    pub params: Vec<KernelParam>,
    /// The kernel body.
    pub body: Vec<KernelOp>,
    /// Raw Rust source of the body (temporary — used for string-based MSL/WGSL
    /// emission until Phase 2 populates `body` with real KernelOps).
    pub body_source: Option<String>,
    /// Next unused register number — the allocator's high-water mark.
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

/// Slots with texel access — `&Texture2D` (read-only) and `&mut Texture2D`
/// (read-write) alike. These become storage images (MSL `access::read` /
/// `access::read_write`, SPIR-V `sampled=2`) so that `texture_load_2d`
/// against them lowers to a storage read rather than an invalid sampled
/// fetch. The slot number is the positional param index shared across the
/// buffer/texture/constant namespace.
pub fn storage_texture_slots(kernel: &KernelDef) -> std::collections::BTreeSet<u32> {
    let mut set = std::collections::BTreeSet::new();
    for p in &kernel.params {
        if let KernelParam::Texture2DRead { slot, .. }
        | KernelParam::Texture2DReadWrite { slot, .. } = p
        {
            set.insert(*slot);
        }
    }
    set
}

/// Slots declared `&Texture2D` (read-only texel). The write guard keys on
/// this set, and emitters mark these NonWritable / `access::read`.
pub fn read_only_texture_slots(kernel: &KernelDef) -> std::collections::BTreeSet<u32> {
    let mut set = std::collections::BTreeSet::new();
    for p in &kernel.params {
        if let KernelParam::Texture2DRead { slot, .. } = p {
            set.insert(*slot);
        }
    }
    set
}

fn walk_ops(ops: &[KernelOp], hit: &mut impl FnMut(&KernelOp) -> Option<u32>) -> Result<(), u32> {
    for op in ops {
        if let Some(slot) = hit(op) {
            return Err(slot);
        }
        match op {
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => {
                walk_ops(then_ops, hit)?;
                walk_ops(else_ops, hit)?;
            }
            KernelOp::Loop { body, .. } => walk_ops(body, hit)?,
            _ => {}
        }
    }
    Ok(())
}

fn walk_kernel(
    kernel: &KernelDef,
    hit: &mut impl FnMut(&KernelOp) -> Option<u32>,
) -> Result<(), u32> {
    walk_ops(&kernel.body, hit)?;
    for f in &kernel.device_functions {
        walk_ops(&f.body, hit)?;
    }
    Ok(())
}

/// Reject `texture_sample_2d`/`texture_sample_3d` against a texel-declared
/// slot. Sampling needs a sampled image with a bound sampler; a storage image
/// (which is what both `&Texture2D` and `&mut Texture2D` become) cannot be
/// sampled — reading it is `texture_load_2d`. Every emitter and the CPU
/// driver call this so all backends agree instead of failing later as
/// undeclared-identifier MSL / invalid SPIR-V.
pub fn reject_sample_on_storage(kernel: &KernelDef) -> Result<(), String> {
    let storage = storage_texture_slots(kernel);
    if storage.is_empty() {
        return Ok(());
    }
    let hit = &mut |op: &KernelOp| match op {
        KernelOp::TextureSample2D { texture, .. } | KernelOp::TextureSample3D { texture, .. }
            if storage.contains(texture) =>
        {
            Some(*texture)
        }
        _ => None,
    };
    if let Err(slot) = walk_kernel(kernel, hit) {
        return Err(format!(
            "texture slot {slot} is declared `Texture2D` (texel access) but is sampled: \
             a storage image cannot be sampled. Use texture_load_2d to read it, or \
             declare the parameter `&Sampled2D` for sampling."
        ));
    }
    Ok(())
}

/// Reject `texture_write_2d` against a read-only texel slot. `&Texture2D` is
/// the read-only half of the texel lattice — its storage image is declared
/// NonWritable / `access::read`, so a write would be invalid SPIR-V / MSL.
/// Every emitter and the CPU driver call this so all backends agree.
pub fn reject_write_on_read_only(kernel: &KernelDef) -> Result<(), String> {
    let read_only = read_only_texture_slots(kernel);
    if read_only.is_empty() {
        return Ok(());
    }
    let hit = &mut |op: &KernelOp| match op {
        KernelOp::TextureWrite2D { texture, .. } if read_only.contains(texture) => Some(*texture),
        _ => None,
    };
    if let Err(slot) = walk_kernel(kernel, hit) {
        return Err(format!(
            "texture slot {slot} is declared `&Texture2D` (read-only texel) but is \
             written: declare it `&mut Texture2D` for read-write access."
        ));
    }
    Ok(())
}

/// Reject a sampled `&Sampled2D<u32>` param. In texel position `u32` means
/// the packed RGBA8-unorm image (`&Texture2D<u32>` / `&mut Texture2D<u32>`);
/// a *sampled* u32 texture would mean something else — an unsigned-integer
/// sampled image (R32Uint / RGBA8Uint) — which is not wired. Rather than
/// silently emit it as a float sampled image (`emit_sampled_2d` ignores the
/// scalar), refuse it at emit so the meaning of sampled u32 stays free for a
/// future arc. Every SPIR-V emitter calls this so both backends agree.
pub fn reject_sampled_u32_texture(kernel: &KernelDef) -> Result<(), String> {
    for p in &kernel.params {
        if let KernelParam::Sampled2D {
            slot,
            scalar_type: ScalarType::U32,
            ..
        } = p
        {
            return Err(format!(
                "texture slot {slot} is a sampled `&Sampled2D<u32>`, which is not supported: \
                 a sampled unsigned-integer texture is a distinct, unwired meaning. Use \
                 `&Sampled2D<f32>` to sample, or `&Texture2D<u32>` / `&mut Texture2D<u32>` \
                 for packed-RGBA8 texel access."
            ));
        }
    }
    Ok(())
}

impl ScalarType {
    /// SPIR-V `ImageFormat` operand for a storage image whose texel is this
    /// scalar. The format contract is scalar-driven:
    ///
    /// - `Texture2D<f32>` ⇔ R32Float, `ImageFormat = 3` (per the SPIR-V spec
    ///   enum — R32f is 3, not the whole-vector Rgba32f which is 1). R32f needs
    ///   no capability beyond `Shader`.
    /// - `Texture2D<u32>` ⇔ **RGBA8-unorm packed-u32**, `ImageFormat = 4`
    ///   (Rgba8, one past R32f in the enum). In storage-texture position `u32`
    ///   deliberately means the packed RGBA8 image, *not* R32Uint: the texel
    ///   crosses the kernel boundary as a `0xAABBGGRR` u32, packed/unpacked to
    ///   a `vec4<f32>` at the OpImageRead/OpImageWrite via
    ///   Pack/UnpackUnorm4x8. The image's SPIR-V sampled type is therefore
    ///   still f32 for both — only this format word differs. Rgba8 storage
    ///   is a mandatory format, so it also needs no capability beyond `Shader`.
    ///
    /// Any other scalar has no wired storage format and is refused at the call
    /// site.
    pub fn spirv_storage_image_format(&self) -> Option<u32> {
        match self {
            Self::F32 => Some(3), // R32f
            Self::U32 => Some(4), // Rgba8 (packed-u32 RGBA8-unorm)
            _ => None,
        }
    }

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
