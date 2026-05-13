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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// `*const T` — read-only buffer.
    BufferRead,
    /// `*mut T` — write or read-write buffer.
    BufferWrite,
    /// `T` — scalar push constant.
    Scalar,
}

#[derive(Debug, Clone)]
pub struct SideTable {
    pub kernel_name: String,
    pub params: Vec<ParamSlot>,
    pub workgroup_size: [u32; 3],
}

#[derive(Debug)]
pub enum LoweringError {
    Parse(String),
    KernelNotFound(String),
    UnsupportedOp { op: String, at: usize },
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

#[derive(Debug, Clone)]
pub struct FnSig {
    pub params: Vec<WasmTy>,
    pub results: Vec<WasmTy>,
}

#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub type_index: u32,
    pub kind: FunctionKind,
}

#[derive(Debug, Clone)]
pub enum FunctionKind {
    /// Imported (e.g. `import "quanta" "quark_id"`).
    Imported { module: String, name: String },
    /// Defined locally — has a body of locals + instructions.
    Defined(FunctionBodyInfo),
}

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

#[derive(Debug, Clone)]
pub struct ImportInfo {
    pub module: String,
    pub name: String,
    pub kind: ImportKind,
}

#[derive(Debug, Clone)]
pub enum ImportKind {
    Function {
        type_index: u32,
    },
    /// Memory / table / global imports — kernels generally don't have
    /// these, but we record them to surface clear errors if a build
    /// emits something we don't expect.
    Other(String),
}

#[derive(Debug, Clone)]
pub struct ExportInfo {
    pub name: String,
    pub kind: ExportKind,
}

#[derive(Debug, Clone)]
pub enum ExportKind {
    Function { index: u32 },
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmTy {
    I32,
    I64,
    F32,
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
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),
    // Constants
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),
    // Integer arithmetic
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rotl,
    I32Rotr,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I32Eqz,
    // Float arithmetic
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,
    F32Neg,
    F32Abs,
    F32Sqrt,
    F32Min,
    F32Max,
    // Conversions
    I32WrapI64,
    F32ConvertI32S,
    F32ConvertI32U,
    I32TruncF32S,
    I32TruncF32U,
    F32ReinterpretI32,
    I32ReinterpretF32,
    // Memory
    I32Load {
        offset: u64,
        align: u32,
    },
    I32Store {
        offset: u64,
        align: u32,
    },
    F32Load {
        offset: u64,
        align: u32,
    },
    F32Store {
        offset: u64,
        align: u32,
    },
    I32Load8U {
        offset: u64,
        align: u32,
    },
    I32Load8S {
        offset: u64,
        align: u32,
    },
    I32Store8 {
        offset: u64,
        align: u32,
    },
    // Control flow
    Block {
        ty_arity: u32,
    },
    Loop {
        ty_arity: u32,
    },
    If {
        ty_arity: u32,
    },
    Else,
    End,
    Br(u32),
    BrIf(u32),
    Return,
    // Calls
    Call(u32),
    // Misc
    Drop,
    Select,
    Unreachable,
    Nop,
    /// Anything we haven't enumerated yet — captured so the lowering
    /// pass can produce a precise "this op isn't supported" error
    /// pointing at the original Rust source via WASM debug info.
    Unsupported(&'static str),
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

            Operator::I32WrapI64 => Self::I32WrapI64,
            Operator::F32ConvertI32S => Self::F32ConvertI32S,
            Operator::F32ConvertI32U => Self::F32ConvertI32U,
            Operator::I32TruncF32S => Self::I32TruncF32S,
            Operator::I32TruncF32U => Self::I32TruncF32U,
            Operator::F32ReinterpretI32 => Self::F32ReinterpretI32,
            Operator::I32ReinterpretF32 => Self::I32ReinterpretF32,

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

fn operator_name(_op: &Operator<'_>) -> &'static str {
    // Best-effort: just say "unknown" for now. We can add a richer
    // mapping when an op-class actually surfaces in real kernels.
    "unsupported"
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
