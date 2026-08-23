//! WASM → Quanta KernelOps lowering pass.
//!
//! Consumes the WASM module emitted when a Quanta kernel is built for
//! `wasm32-unknown-unknown` and produces a `KernelDef` the existing
//! emitters (PTX / GCN / SPIR-V / MSL / WGSL) lower to backend ISAs.
//!
//! This commit ships the parser layer:
//! - `parse_module(wasm)` walks the WASM, extracts imports, function
//!   signatures, exports, and per-function (locals + instruction
//!   list).
//! - `find_kernel(module, name)` locates a kernel by its export name.
//! - The op-by-op lowering itself (WASM → `KernelOp` translation) is
//!   the next commit.

#![allow(dead_code)]
#![deny(missing_docs)]

use quanta_ir::{KernelDef, ScalarType};
use wasmparser::{
    ExternalKind, FuncType, FunctionBody, KnownCustom, Name, Operator, Parser, Payload, ValType,
};

// ── Public types ───────────────────────────────────────────────────────

/// Per-parameter metadata the kernel macro emits alongside the WASM,
/// telling the lowering pass which function parameters are buffer
/// pointers vs scalar push constants.
///
/// In the current commit this is supplied by the caller; once the
/// macro-side emitter lands it'll come from a custom WASM section the
/// macro injects.
#[derive(Debug, Clone)]
pub struct ParamSlot {
    /// Position in the WASM function signature.
    pub wasm_index: u32,
    /// Logical slot in the produced `KernelDef.params`.
    pub slot: u32,
    /// Whether this parameter is a buffer pointer or a scalar push
    /// constant.
    pub kind: ParamKind,
    /// Element type (for buffers) or scalar type (for push constants).
    pub scalar: ScalarType,
}

/// Which binding a kernel parameter lowers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// `*const T` — read-only buffer.
    BufferRead,
    /// `*mut T` — write or read-write buffer.
    BufferWrite,
    /// `T` — scalar push constant.
    Scalar,
}

/// Everything the lowering pass needs about a kernel that the WASM
/// itself doesn't carry: which export to lower and how its parameters
/// bind.
#[derive(Debug, Clone)]
pub struct SideTable {
    /// Export name of the kernel function in the WASM module.
    pub kernel_name: String,
    /// One entry per WASM function parameter, in signature order.
    pub params: Vec<ParamSlot>,
    /// Workgroup dimensions the kernel was declared with.
    pub workgroup_size: [u32; 3],
}

/// Why a WASM module could not be turned into a `KernelDef`.
#[derive(Debug)]
pub enum LoweringError {
    /// The WASM binary could not be decoded.
    Parse(String),
    /// No export in the module carries the requested kernel name.
    KernelNotFound(String),
    /// The body reached an instruction this pass does not translate.
    UnsupportedOp {
        /// The instruction's name, as wasmparser spells it.
        op: String,
        /// Byte offset of the instruction within the module.
        at: usize,
    },
    /// The side table and the WASM signature describe different kernels.
    ShapeMismatch(String),
}

impl core::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Parse(s) => write!(f, "wasm parse error: {s}"),
            Self::KernelNotFound(name) => write!(f, "kernel `{name}` not found in WASM module"),
            Self::UnsupportedOp { op, at } => {
                write!(f, "unsupported WASM op `{op}` at offset {at}")
            }
            Self::ShapeMismatch(s) => write!(f, "wasm/side-table shape mismatch: {s}"),
        }
    }
}

impl core::error::Error for LoweringError {}

impl From<wasmparser::BinaryReaderError> for LoweringError {
    fn from(e: wasmparser::BinaryReaderError) -> Self {
        Self::Parse(format!("{e}"))
    }
}

// ── Module representation ──────────────────────────────────────────────

/// Decoded view of a WASM module — only the bits the lowering pass
/// needs. Owns its data so callers don't have to keep the original
/// bytes alive (the wasmparser `Operator` references are cloned via
/// `into_owned` form).
#[derive(Debug, Clone)]
pub struct Module {
    /// Indexed by `type_index`; the function-typed types from the
    /// `Type` section.
    pub types: Vec<FnSig>,
    /// One entry per function, in WASM index order (imports first,
    /// then defined functions).
    pub functions: Vec<FunctionInfo>,
    /// Imported names (for resolving intrinsic call sites — `quark_id`,
    /// `barrier`, etc.).
    pub imports: Vec<ImportInfo>,
    /// Exported names (for finding kernels).
    pub exports: Vec<ExportInfo>,
    /// Function debug names from the WASM `name` custom section,
    /// indexed by WASM function index. `None` entries mean rustc
    /// didn't emit a name for that index. Used to surface meaningful
    /// errors when the lowering pass hits an unsupported defined-
    /// function call (e.g. a stdlib helper).
    pub function_names: Vec<Option<String>>,
}

/// A WASM function type.
#[derive(Debug, Clone)]
pub struct FnSig {
    /// Parameter types, in signature order.
    pub params: Vec<WasmTy>,
    /// Result types, in return order.
    pub results: Vec<WasmTy>,
}

/// One WASM function: its signature, plus where its body comes from.
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// Index into `Module::types`.
    pub type_index: u32,
    /// Imported from outside the module, or defined inside it.
    pub kind: FunctionKind,
}

/// Where a function's body lives — outside the module, or in it.
#[derive(Debug, Clone)]
pub enum FunctionKind {
    /// Imported (e.g. `import "quanta" "quark_id"`).
    Imported {
        /// The import's module name (`quanta` for the kernel intrinsics).
        module: String,
        /// The imported function's name.
        name: String,
    },
    /// Defined locally — has a body of locals + instructions.
    Defined(FunctionBodyInfo),
}

/// A defined function's body: the locals it declares and the
/// instruction stream that follows them.
#[derive(Debug, Clone)]
pub struct FunctionBodyInfo {
    /// Local declarations beyond parameters: `(count, type)` pairs as
    /// WASM stores them. Resolve via `expand_locals`.
    pub locals: Vec<(u32, WasmTy)>,
    /// Linear instruction stream. Each `RawInstr` is an owned copy of
    /// the wasmparser `Operator`'s observable shape.
    pub instructions: Vec<RawInstr>,
    /// Byte offset of the function body for source-mapping.
    pub body_offset: usize,
}

/// One entry of the WASM import section.
#[derive(Debug, Clone)]
pub struct ImportInfo {
    /// The import's module name.
    pub module: String,
    /// The imported item's name.
    pub name: String,
    /// What kind of item it brings in.
    pub kind: ImportKind,
}

/// What an import section entry brings into the module.
#[derive(Debug, Clone)]
pub enum ImportKind {
    /// A function import — the intrinsic call sites lowering resolves.
    Function {
        /// Index into `Module::types`.
        type_index: u32,
    },
    /// Memory / table / global imports — kernels generally don't have
    /// these, but we record them to surface clear errors if a build
    /// emits something we don't expect.
    Other(String),
}

/// One entry of the WASM export section.
#[derive(Debug, Clone)]
pub struct ExportInfo {
    /// The exported name — what `find_kernel` matches against.
    pub name: String,
    /// What kind of item is published under that name.
    pub kind: ExportKind,
}

/// What an export section entry publishes.
#[derive(Debug, Clone)]
pub enum ExportKind {
    /// A function export — the shape every kernel entry point has.
    Function {
        /// WASM index of the exported function.
        index: u32,
    },
    /// A memory / table / global export; the string is its kind.
    Other(String),
}

/// The WASM value types a Quanta kernel body can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmTy {
    /// 32-bit integer — also the pointer width under wasm32.
    I32,
    /// 64-bit integer.
    I64,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
}

impl WasmTy {
    fn from_val(v: ValType) -> Result<Self, LoweringError> {
        match v {
            ValType::I32 => Ok(Self::I32),
            ValType::I64 => Ok(Self::I64),
            ValType::F32 => Ok(Self::F32),
            ValType::F64 => Ok(Self::F64),
            other => Err(LoweringError::Parse(format!(
                "unsupported WASM val type {other:?}"
            ))),
        }
    }
}

/// Owned form of `wasmparser::Operator` for the subset we care about.
/// Adding a new WASM op the lowering pass should handle means adding
/// a variant here AND a match arm in `from_operator`. Anything we
/// don't recognize becomes `Unsupported(name)` — surfaces a clean
/// error at lowering time.
#[derive(Debug, Clone)]
pub enum RawInstr {
    // Locals
    /// `local.get` — push the value of that local.
    LocalGet(u32),
    /// `local.set` — pop the stack top into that local.
    LocalSet(u32),
    /// `local.tee` — write that local, leaving the value on the stack.
    LocalTee(u32),
    // Constants
    /// `i32.const` — push the immediate.
    I32Const(i32),
    /// `i64.const` — push the immediate.
    I64Const(i64),
    /// `f32.const` — push the immediate.
    F32Const(f32),
    /// `f64.const` — push the immediate.
    F64Const(f64),
    // Integer arithmetic
    /// `i32.add`.
    I32Add,
    /// `i32.sub`.
    I32Sub,
    /// `i32.mul`.
    I32Mul,
    /// `i32.div_s` — signed division.
    I32DivS,
    /// `i32.div_u` — unsigned division.
    I32DivU,
    /// `i32.rem_s` — signed remainder.
    I32RemS,
    /// `i32.rem_u` — unsigned remainder.
    I32RemU,
    /// `i32.and`.
    I32And,
    /// `i32.or`.
    I32Or,
    /// `i32.xor`.
    I32Xor,
    /// `i32.shl` — left shift.
    I32Shl,
    /// `i32.shr_s` — arithmetic right shift.
    I32ShrS,
    /// `i32.shr_u` — logical right shift.
    I32ShrU,
    /// `i32.rotl` — rotate left.
    I32Rotl,
    /// `i32.rotr` — rotate right.
    I32Rotr,
    /// `i32.eq`.
    I32Eq,
    /// `i32.ne`.
    I32Ne,
    /// `i32.lt_s` — signed `<`.
    I32LtS,
    /// `i32.lt_u` — unsigned `<`.
    I32LtU,
    /// `i32.gt_s` — signed `>`.
    I32GtS,
    /// `i32.gt_u` — unsigned `>`.
    I32GtU,
    /// `i32.le_s` — signed `<=`.
    I32LeS,
    /// `i32.le_u` — unsigned `<=`.
    I32LeU,
    /// `i32.ge_s` — signed `>=`.
    I32GeS,
    /// `i32.ge_u` — unsigned `>=`.
    I32GeU,
    /// `i32.eqz` — 1 when the operand is zero, else 0.
    I32Eqz,
    // i64 arithmetic (mirrors the i32 surface; lowered with the
    // u64 width class).
    /// `i64.add`.
    I64Add,
    /// `i64.sub`.
    I64Sub,
    /// `i64.mul`.
    I64Mul,
    /// `i64.div_s` — signed division.
    I64DivS,
    /// `i64.div_u` — unsigned division.
    I64DivU,
    /// `i64.rem_s` — signed remainder.
    I64RemS,
    /// `i64.rem_u` — unsigned remainder.
    I64RemU,
    /// `i64.and`.
    I64And,
    /// `i64.or`.
    I64Or,
    /// `i64.xor`.
    I64Xor,
    /// `i64.shl` — left shift.
    I64Shl,
    /// `i64.shr_s` — arithmetic right shift.
    I64ShrS,
    /// `i64.shr_u` — logical right shift.
    I64ShrU,
    /// `i64.rotl` — rotate left.
    I64Rotl,
    /// `i64.rotr` — rotate right.
    I64Rotr,
    /// `i64.eq`.
    I64Eq,
    /// `i64.ne`.
    I64Ne,
    /// `i64.lt_s` — signed `<`.
    I64LtS,
    /// `i64.lt_u` — unsigned `<`.
    I64LtU,
    /// `i64.gt_s` — signed `>`.
    I64GtS,
    /// `i64.gt_u` — unsigned `>`.
    I64GtU,
    /// `i64.le_s` — signed `<=`.
    I64LeS,
    /// `i64.le_u` — unsigned `<=`.
    I64LeU,
    /// `i64.ge_s` — signed `>=`.
    I64GeS,
    /// `i64.ge_u` — unsigned `>=`.
    I64GeU,
    /// `i64.eqz` — 1 when the operand is zero, else 0.
    I64Eqz,
    // Float arithmetic
    /// `f32.add`.
    F32Add,
    /// `f32.sub`.
    F32Sub,
    /// `f32.mul`.
    F32Mul,
    /// `f32.div`.
    F32Div,
    /// `f32.eq`.
    F32Eq,
    /// `f32.ne`.
    F32Ne,
    /// `f32.lt`.
    F32Lt,
    /// `f32.gt`.
    F32Gt,
    /// `f32.le`.
    F32Le,
    /// `f32.ge`.
    F32Ge,
    /// `f32.neg`.
    F32Neg,
    /// `f32.abs`.
    F32Abs,
    /// `f32.sqrt`.
    F32Sqrt,
    /// `f32.min`.
    F32Min,
    /// `f32.max`.
    F32Max,
    // f64 arithmetic — mirrors the f32 surface above. Lowered with
    // the F64 width class.
    /// `f64.add`.
    F64Add,
    /// `f64.sub`.
    F64Sub,
    /// `f64.mul`.
    F64Mul,
    /// `f64.div`.
    F64Div,
    /// `f64.eq`.
    F64Eq,
    /// `f64.ne`.
    F64Ne,
    /// `f64.lt`.
    F64Lt,
    /// `f64.gt`.
    F64Gt,
    /// `f64.le`.
    F64Le,
    /// `f64.ge`.
    F64Ge,
    /// `f64.neg`.
    F64Neg,
    /// `f64.abs`.
    F64Abs,
    /// `f64.sqrt`.
    F64Sqrt,
    /// `f64.min`.
    F64Min,
    /// `f64.max`.
    F64Max,
    // Conversions
    /// `i32.wrap_i64` — keep the low 32 bits.
    I32WrapI64,
    /// `i64.extend_i32_s` — sign-extend.
    I64ExtendI32S,
    /// `i64.extend_i32_u` — zero-extend.
    I64ExtendI32U,
    /// `f32.convert_i32_s` — signed int → float.
    F32ConvertI32S,
    /// `f32.convert_i32_u` — unsigned int → float.
    F32ConvertI32U,
    /// `i32.trunc_f32_s` — float → signed int, toward zero.
    I32TruncF32S,
    /// `i32.trunc_f32_u` — float → unsigned int, toward zero.
    I32TruncF32U,
    /// `f32.reinterpret_i32` — bit cast.
    F32ReinterpretI32,
    /// `i32.reinterpret_f32` — bit cast.
    I32ReinterpretF32,
    // f32 ↔ f64 width conversions.
    /// `f64.promote_f32` — widen.
    F64PromoteF32,
    /// `f32.demote_f64` — narrow, rounding to nearest.
    F32DemoteF64,
    // f64 ↔ int conversions.
    /// `f64.convert_i32_s` — signed int → float.
    F64ConvertI32S,
    /// `f64.convert_i32_u` — unsigned int → float.
    F64ConvertI32U,
    /// `f64.convert_i64_s` — signed int → float.
    F64ConvertI64S,
    /// `f64.convert_i64_u` — unsigned int → float.
    F64ConvertI64U,
    /// `i32.trunc_f64_s` — float → signed int, toward zero.
    I32TruncF64S,
    /// `i32.trunc_f64_u` — float → unsigned int, toward zero.
    I32TruncF64U,
    /// `i64.trunc_f64_s` — float → signed int, toward zero.
    I64TruncF64S,
    /// `i64.trunc_f64_u` — float → unsigned int, toward zero.
    I64TruncF64U,
    // Saturating float→int trunc (WASM 2.0 `nontrapping-fptoint`
    // proposal). Rustc emits these for `as` casts between f32/f64
    // and integer types when the proposal is enabled (the default
    // since 2020). Lowered identically to the baseline `TruncF*`
    // variants — `KernelOp::Cast` carries saturating semantics in
    // every backend (CPU eval uses Rust's saturating `as`, MSL
    // uses `int(min(max(x,…),…))`, SPIR-V uses
    // `OpConvertFToU/S`+clamp).
    /// `i32.trunc_sat_f32_s` — float → signed int, toward zero, saturating.
    I32TruncSatF32S,
    /// `i32.trunc_sat_f32_u` — float → unsigned int, toward zero, saturating.
    I32TruncSatF32U,
    /// `i32.trunc_sat_f64_s` — float → signed int, toward zero, saturating.
    I32TruncSatF64S,
    /// `i32.trunc_sat_f64_u` — float → unsigned int, toward zero, saturating.
    I32TruncSatF64U,
    /// `i64.trunc_sat_f32_s` — float → signed int, toward zero, saturating.
    I64TruncSatF32S,
    /// `i64.trunc_sat_f32_u` — float → unsigned int, toward zero, saturating.
    I64TruncSatF32U,
    /// `i64.trunc_sat_f64_s` — float → signed int, toward zero, saturating.
    I64TruncSatF64S,
    /// `i64.trunc_sat_f64_u` — float → unsigned int, toward zero, saturating.
    I64TruncSatF64U,
    /// `f64.reinterpret_i64` — bit cast.
    F64ReinterpretI64,
    /// `i64.reinterpret_f64` — bit cast.
    I64ReinterpretF64,
    // Memory
    /// `i32.load`.
    I32Load {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i32.store`.
    I32Store {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `f32.load`.
    F32Load {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `f32.store`.
    F32Store {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `f64.load`.
    F64Load {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `f64.store`.
    F64Store {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i32.load8_u` — load one byte, zero-extended.
    I32Load8U {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i32.load8_s` — load one byte, sign-extended.
    I32Load8S {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i32.store8` — store the low byte.
    I32Store8 {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    // i64 memory ops. `I64Load` / `I64Store` read/write 8-byte
    // values from a u64/i64 buffer slot. The narrow variants
    // (Load32U / Load32S / Store32) mirror the i32 narrow load/
    // store family — used when rustc fuses `(u32_buf[i] as u64)`
    // into a single load+widen instruction.
    /// `i64.load`.
    I64Load {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.store`.
    I64Store {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.load32_u` — load four bytes, zero-extended.
    I64Load32U {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.load32_s` — load four bytes, sign-extended.
    I64Load32S {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.load16_u` — load two bytes, zero-extended.
    I64Load16U {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.load16_s` — load two bytes, sign-extended.
    I64Load16S {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.load8_u` — load one byte, zero-extended.
    I64Load8U {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.load8_s` — load one byte, sign-extended.
    I64Load8S {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.store32` — store the low four bytes.
    I64Store32 {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.store16` — store the low two bytes.
    I64Store16 {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    /// `i64.store8` — store the low byte.
    I64Store8 {
        /// Static byte offset folded into the computed address.
        offset: u64,
        /// Alignment hint — log2 of the expected byte alignment.
        align: u32,
    },
    // Control flow
    /// `block` — a label branched to at its end.
    Block {
        /// Number of result values the block type yields.
        ty_arity: u32,
    },
    /// `loop` — a label branched to at its start.
    Loop {
        /// Number of result values the block type yields.
        ty_arity: u32,
    },
    /// `if` — enter the consequent when the popped condition is non-zero.
    If {
        /// Number of result values the block type yields.
        ty_arity: u32,
    },
    /// `else` — the alternative arm of the enclosing `if`.
    Else,
    /// `end` — close the innermost block, loop, if, or function body.
    End,
    /// `br` — branch out that many label levels.
    Br(u32),
    /// `br_if` — the same branch, taken when the popped condition
    /// is non-zero.
    BrIf(u32),
    /// `return` — leave the function with the current stack results.
    Return,
    // Calls
    /// `call` — invoke the function at that WASM index.
    Call(u32),
    // Misc
    /// `drop` — pop and discard the stack top.
    Drop,
    /// `select` — pop a condition and two values, keeping the first
    /// when the condition is non-zero.
    Select,
    /// `unreachable` — trap.
    Unreachable,
    /// `nop` — no operation.
    Nop,
    /// Anything we haven't enumerated yet — captured so the lowering
    /// pass can produce a precise "this op isn't supported" error
    /// pointing at the original Rust source via WASM debug info.
    /// String form is the wasmparser `Operator` Debug variant name
    /// (truncated to the head of `format!("{:?}", op)`) — enough to
    /// tell `Br` apart from `F32x4Add` at a glance.
    Unsupported(String),
}

impl RawInstr {
    fn from_operator(op: &Operator<'_>) -> Self {
        match op {
            Operator::LocalGet { local_index } => Self::LocalGet(*local_index),
            Operator::LocalSet { local_index } => Self::LocalSet(*local_index),
            Operator::LocalTee { local_index } => Self::LocalTee(*local_index),

            Operator::I32Const { value } => Self::I32Const(*value),
            Operator::I64Const { value } => Self::I64Const(*value),
            Operator::F32Const { value } => Self::F32Const(f32::from_bits(value.bits())),
            Operator::F64Const { value } => Self::F64Const(f64::from_bits(value.bits())),

            Operator::I32Add => Self::I32Add,
            Operator::I32Sub => Self::I32Sub,
            Operator::I32Mul => Self::I32Mul,
            Operator::I32DivS => Self::I32DivS,
            Operator::I32DivU => Self::I32DivU,
            Operator::I32RemS => Self::I32RemS,
            Operator::I32RemU => Self::I32RemU,
            Operator::I32And => Self::I32And,
            Operator::I32Or => Self::I32Or,
            Operator::I32Xor => Self::I32Xor,
            Operator::I32Shl => Self::I32Shl,
            Operator::I32ShrS => Self::I32ShrS,
            Operator::I32ShrU => Self::I32ShrU,
            Operator::I32Rotl => Self::I32Rotl,
            Operator::I32Rotr => Self::I32Rotr,
            Operator::I32Eq => Self::I32Eq,
            Operator::I32Ne => Self::I32Ne,
            Operator::I32LtS => Self::I32LtS,
            Operator::I32LtU => Self::I32LtU,
            Operator::I32GtS => Self::I32GtS,
            Operator::I32GtU => Self::I32GtU,
            Operator::I32LeS => Self::I32LeS,
            Operator::I32LeU => Self::I32LeU,
            Operator::I32GeS => Self::I32GeS,
            Operator::I32GeU => Self::I32GeU,
            Operator::I32Eqz => Self::I32Eqz,

            Operator::I64Add => Self::I64Add,
            Operator::I64Sub => Self::I64Sub,
            Operator::I64Mul => Self::I64Mul,
            Operator::I64DivS => Self::I64DivS,
            Operator::I64DivU => Self::I64DivU,
            Operator::I64RemS => Self::I64RemS,
            Operator::I64RemU => Self::I64RemU,
            Operator::I64And => Self::I64And,
            Operator::I64Or => Self::I64Or,
            Operator::I64Xor => Self::I64Xor,
            Operator::I64Shl => Self::I64Shl,
            Operator::I64ShrS => Self::I64ShrS,
            Operator::I64ShrU => Self::I64ShrU,
            Operator::I64Rotl => Self::I64Rotl,
            Operator::I64Rotr => Self::I64Rotr,
            Operator::I64Eq => Self::I64Eq,
            Operator::I64Ne => Self::I64Ne,
            Operator::I64LtS => Self::I64LtS,
            Operator::I64LtU => Self::I64LtU,
            Operator::I64GtS => Self::I64GtS,
            Operator::I64GtU => Self::I64GtU,
            Operator::I64LeS => Self::I64LeS,
            Operator::I64LeU => Self::I64LeU,
            Operator::I64GeS => Self::I64GeS,
            Operator::I64GeU => Self::I64GeU,
            Operator::I64Eqz => Self::I64Eqz,

            Operator::F32Add => Self::F32Add,
            Operator::F32Sub => Self::F32Sub,
            Operator::F32Mul => Self::F32Mul,
            Operator::F32Div => Self::F32Div,
            Operator::F32Eq => Self::F32Eq,
            Operator::F32Ne => Self::F32Ne,
            Operator::F32Lt => Self::F32Lt,
            Operator::F32Gt => Self::F32Gt,
            Operator::F32Le => Self::F32Le,
            Operator::F32Ge => Self::F32Ge,
            Operator::F32Neg => Self::F32Neg,
            Operator::F32Abs => Self::F32Abs,
            Operator::F32Sqrt => Self::F32Sqrt,
            Operator::F32Min => Self::F32Min,
            Operator::F32Max => Self::F32Max,

            Operator::F64Add => Self::F64Add,
            Operator::F64Sub => Self::F64Sub,
            Operator::F64Mul => Self::F64Mul,
            Operator::F64Div => Self::F64Div,
            Operator::F64Eq => Self::F64Eq,
            Operator::F64Ne => Self::F64Ne,
            Operator::F64Lt => Self::F64Lt,
            Operator::F64Gt => Self::F64Gt,
            Operator::F64Le => Self::F64Le,
            Operator::F64Ge => Self::F64Ge,
            Operator::F64Neg => Self::F64Neg,
            Operator::F64Abs => Self::F64Abs,
            Operator::F64Sqrt => Self::F64Sqrt,
            Operator::F64Min => Self::F64Min,
            Operator::F64Max => Self::F64Max,

            Operator::I32WrapI64 => Self::I32WrapI64,
            Operator::I64ExtendI32S => Self::I64ExtendI32S,
            Operator::I64ExtendI32U => Self::I64ExtendI32U,
            Operator::F32ConvertI32S => Self::F32ConvertI32S,
            Operator::F32ConvertI32U => Self::F32ConvertI32U,
            Operator::I32TruncF32S => Self::I32TruncF32S,
            Operator::I32TruncF32U => Self::I32TruncF32U,
            Operator::F32ReinterpretI32 => Self::F32ReinterpretI32,
            Operator::I32ReinterpretF32 => Self::I32ReinterpretF32,

            Operator::F64PromoteF32 => Self::F64PromoteF32,
            Operator::F32DemoteF64 => Self::F32DemoteF64,
            Operator::F64ConvertI32S => Self::F64ConvertI32S,
            Operator::F64ConvertI32U => Self::F64ConvertI32U,
            Operator::F64ConvertI64S => Self::F64ConvertI64S,
            Operator::F64ConvertI64U => Self::F64ConvertI64U,
            Operator::I32TruncF64S => Self::I32TruncF64S,
            Operator::I32TruncF64U => Self::I32TruncF64U,
            Operator::I64TruncF64S => Self::I64TruncF64S,
            Operator::I64TruncF64U => Self::I64TruncF64U,
            Operator::I32TruncSatF32S => Self::I32TruncSatF32S,
            Operator::I32TruncSatF32U => Self::I32TruncSatF32U,
            Operator::I32TruncSatF64S => Self::I32TruncSatF64S,
            Operator::I32TruncSatF64U => Self::I32TruncSatF64U,
            Operator::I64TruncSatF32S => Self::I64TruncSatF32S,
            Operator::I64TruncSatF32U => Self::I64TruncSatF32U,
            Operator::I64TruncSatF64S => Self::I64TruncSatF64S,
            Operator::I64TruncSatF64U => Self::I64TruncSatF64U,
            Operator::F64ReinterpretI64 => Self::F64ReinterpretI64,
            Operator::I64ReinterpretF64 => Self::I64ReinterpretF64,

            Operator::I32Load { memarg } => Self::I32Load {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I32Store { memarg } => Self::I32Store {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::F32Load { memarg } => Self::F32Load {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::F32Store { memarg } => Self::F32Store {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::F64Load { memarg } => Self::F64Load {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::F64Store { memarg } => Self::F64Store {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I32Load8U { memarg } => Self::I32Load8U {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I32Load8S { memarg } => Self::I32Load8S {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I32Store8 { memarg } => Self::I32Store8 {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Load { memarg } => Self::I64Load {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Store { memarg } => Self::I64Store {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Load32U { memarg } => Self::I64Load32U {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Load32S { memarg } => Self::I64Load32S {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Store32 { memarg } => Self::I64Store32 {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Load16U { memarg } => Self::I64Load16U {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Load16S { memarg } => Self::I64Load16S {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Load8U { memarg } => Self::I64Load8U {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Load8S { memarg } => Self::I64Load8S {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Store16 { memarg } => Self::I64Store16 {
                offset: memarg.offset,
                align: memarg.align as u32,
            },
            Operator::I64Store8 { memarg } => Self::I64Store8 {
                offset: memarg.offset,
                align: memarg.align as u32,
            },

            // Block-typed ops: store the type arity for now (full
            // block-type encoding is more complex and not yet needed).
            Operator::Block { .. } => Self::Block { ty_arity: 0 },
            Operator::Loop { .. } => Self::Loop { ty_arity: 0 },
            Operator::If { .. } => Self::If { ty_arity: 0 },
            Operator::Else => Self::Else,
            Operator::End => Self::End,
            Operator::Br { relative_depth } => Self::Br(*relative_depth),
            Operator::BrIf { relative_depth } => Self::BrIf(*relative_depth),
            Operator::Return => Self::Return,

            Operator::Call { function_index } => Self::Call(*function_index),

            Operator::Drop => Self::Drop,
            Operator::Select => Self::Select,
            Operator::Unreachable => Self::Unreachable,
            Operator::Nop => Self::Nop,

            other => Self::Unsupported(operator_name(other)),
        }
    }
}

fn operator_name(op: &Operator<'_>) -> String {
    // wasmparser's `Operator` doesn't implement `Display`; the Debug
    // form is `Variant { field: ... }` for ops with operands and
    // `Variant` otherwise. Truncate to the first non-alphanumeric to
    // grab just the variant name — that's the diagnostic signal we
    // need ("F32x4RelaxedMadd" tells us we hit a relaxed-SIMD op,
    // "TryTable" tells us we hit exception handling, etc.). The
    // string is then surfaced verbatim by `LoweringError::UnsupportedOp`.
    let dbg = format!("{op:?}");
    let head: String = dbg
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if head.is_empty() { dbg } else { head }
}

// ── Parser entry point ─────────────────────────────────────────────────

/// Parse a WASM module into the `Module` representation.
pub fn parse_module(wasm: &[u8]) -> Result<Module, LoweringError> {
    let parser = Parser::new(0);
    let mut types: Vec<FnSig> = Vec::new();
    let mut imports: Vec<ImportInfo> = Vec::new();
    let mut import_func_count: u32 = 0;
    let mut function_type_indices: Vec<u32> = Vec::new();
    let mut function_bodies: Vec<FunctionBodyInfo> = Vec::new();
    let mut exports: Vec<ExportInfo> = Vec::new();
    let mut function_name_pairs: Vec<(u32, String)> = Vec::new();

    for payload in parser.parse_all(wasm) {
        match payload? {
            Payload::TypeSection(reader) => {
                for ty in reader.into_iter_with_offsets() {
                    let (_off, recgroup) = ty?;
                    for sub in recgroup.types() {
                        let comp = &sub.composite_type.inner;
                        if let wasmparser::CompositeInnerType::Func(ft) = comp {
                            types.push(fn_sig_from(ft)?);
                        } else {
                            types.push(FnSig {
                                params: Vec::new(),
                                results: Vec::new(),
                            });
                        }
                    }
                }
            }
            Payload::ImportSection(reader) => {
                for imp in reader {
                    let imp = imp?;
                    let kind = match imp.ty {
                        wasmparser::TypeRef::Func(idx) => {
                            import_func_count += 1;
                            ImportKind::Function { type_index: idx }
                        }
                        wasmparser::TypeRef::Memory(_) => ImportKind::Other("memory".into()),
                        wasmparser::TypeRef::Table(_) => ImportKind::Other("table".into()),
                        wasmparser::TypeRef::Global(_) => ImportKind::Other("global".into()),
                        wasmparser::TypeRef::Tag(_) => ImportKind::Other("tag".into()),
                    };
                    imports.push(ImportInfo {
                        module: imp.module.to_string(),
                        name: imp.name.to_string(),
                        kind,
                    });
                }
            }
            Payload::FunctionSection(reader) => {
                for ty_idx in reader {
                    function_type_indices.push(ty_idx?);
                }
            }
            Payload::ExportSection(reader) => {
                for exp in reader {
                    let exp = exp?;
                    let kind = match exp.kind {
                        ExternalKind::Func => ExportKind::Function { index: exp.index },
                        ExternalKind::Memory => ExportKind::Other("memory".into()),
                        ExternalKind::Table => ExportKind::Other("table".into()),
                        ExternalKind::Global => ExportKind::Other("global".into()),
                        ExternalKind::Tag => ExportKind::Other("tag".into()),
                    };
                    exports.push(ExportInfo {
                        name: exp.name.to_string(),
                        kind,
                    });
                }
            }
            Payload::CodeSectionEntry(body) => {
                function_bodies.push(decode_body(&body)?);
            }
            Payload::CustomSection(reader) => {
                if let KnownCustom::Name(name_reader) = reader.as_known() {
                    for sub in name_reader {
                        let Ok(sub) = sub else { continue };
                        if let Name::Function(map) = sub {
                            for naming in map {
                                let Ok(naming) = naming else { continue };
                                function_name_pairs.push((naming.index, naming.name.to_string()));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Stitch the function table together: imports first (their
    // bodies are `Imported`), then defined functions (with bodies).
    let mut functions = Vec::with_capacity(import_func_count as usize + function_bodies.len());
    let mut import_idx = 0u32;
    for imp in &imports {
        if let ImportKind::Function { type_index } = &imp.kind {
            functions.push(FunctionInfo {
                type_index: *type_index,
                kind: FunctionKind::Imported {
                    module: imp.module.clone(),
                    name: imp.name.clone(),
                },
            });
            import_idx += 1;
        }
    }
    debug_assert_eq!(import_idx, import_func_count);

    for (i, body) in function_bodies.into_iter().enumerate() {
        let type_index = function_type_indices.get(i).copied().ok_or_else(|| {
            LoweringError::Parse(format!("function {i} has no entry in the Function section"))
        })?;
        functions.push(FunctionInfo {
            type_index,
            kind: FunctionKind::Defined(body),
        });
    }

    let mut function_names = vec![None; functions.len()];
    for (idx, name) in function_name_pairs {
        if let Some(slot) = function_names.get_mut(idx as usize) {
            *slot = Some(name);
        }
    }

    Ok(Module {
        types,
        functions,
        imports,
        exports,
        function_names,
    })
}

fn fn_sig_from(ft: &FuncType) -> Result<FnSig, LoweringError> {
    let mut params = Vec::with_capacity(ft.params().len());
    for p in ft.params() {
        params.push(WasmTy::from_val(*p)?);
    }
    let mut results = Vec::with_capacity(ft.results().len());
    for r in ft.results() {
        results.push(WasmTy::from_val(*r)?);
    }
    Ok(FnSig { params, results })
}

fn decode_body(body: &FunctionBody<'_>) -> Result<FunctionBodyInfo, LoweringError> {
    let mut locals = Vec::new();
    for entry in body.get_locals_reader()? {
        let (count, ty) = entry?;
        locals.push((count, WasmTy::from_val(ty)?));
    }
    let mut instructions = Vec::new();
    for op in body.get_operators_reader()? {
        let op = op?;
        instructions.push(RawInstr::from_operator(&op));
    }
    Ok(FunctionBodyInfo {
        locals,
        instructions,
        body_offset: body.range().start,
    })
}

// ── Convenience: locate a kernel by name ───────────────────────────────

/// Find a defined function by its export name. Returns the
/// `FunctionInfo` and its WASM-level function index.
pub fn find_kernel<'a>(
    module: &'a Module,
    export_name: &str,
) -> Result<(u32, &'a FunctionInfo), LoweringError> {
    let exp = module
        .exports
        .iter()
        .find(|e| e.name == export_name)
        .ok_or_else(|| LoweringError::KernelNotFound(export_name.into()))?;
    let index = match exp.kind {
        ExportKind::Function { index } => index,
        _ => {
            return Err(LoweringError::KernelNotFound(format!(
                "{export_name} is not a function export"
            )));
        }
    };
    let info = module
        .functions
        .get(index as usize)
        .ok_or_else(|| LoweringError::KernelNotFound(export_name.into()))?;
    Ok((index, info))
}

// ── Lowering entry point ───────────────────────────────────────────────

mod lower;

/// Lower a WASM module + side table into a `KernelDef`.
///
/// Walks the WASM instruction stream of the kernel named in the side
/// table, simulating the WASM stack machine with a symbolic abstract
/// domain that recognizes the canonical buffer-access patterns rustc
/// emits, and produces an equivalent `KernelOp` list.
pub fn lower(wasm: &[u8], side_table: &SideTable) -> Result<KernelDef, LoweringError> {
    lower::lower_module(wasm, side_table)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A handcrafted WASM module containing one exported `add` function:
    ///   (func (export "add") (param i32 i32) (result i32)
    ///     local.get 0
    ///     local.get 1
    ///     i32.add)
    /// Built from the WASM binary spec — small enough to vendor.
    const ADD_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, // magic
        0x01, 0x00, 0x00, 0x00, // version
        // Type section: one (i32, i32) -> i32
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
        // Function section: function 0 has type 0
        0x03, 0x02, 0x01, 0x00, // Export section: "add" -> func 0
        0x07, 0x07, 0x01, 0x03, b'a', b'd', b'd', 0x00, 0x00,
        // Code section: func 0 body
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
    ];

    #[test]
    fn parses_minimal_add() {
        let module = parse_module(ADD_WASM).unwrap();
        assert_eq!(module.types.len(), 1);
        assert_eq!(module.types[0].params, vec![WasmTy::I32, WasmTy::I32]);
        assert_eq!(module.types[0].results, vec![WasmTy::I32]);
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.exports.len(), 1);
        assert_eq!(module.exports[0].name, "add");

        let (_idx, info) = find_kernel(&module, "add").unwrap();
        match &info.kind {
            FunctionKind::Defined(body) => {
                assert!(body.locals.is_empty());
                assert!(matches!(
                    body.instructions.as_slice(),
                    [
                        RawInstr::LocalGet(0),
                        RawInstr::LocalGet(1),
                        RawInstr::I32Add,
                        RawInstr::End,
                    ]
                ));
            }
            _ => panic!("expected defined function"),
        }
    }
}
