//! WASM → Quanta KernelOps lowering pass.
//!
//! Consumes the WASM module emitted when a Quanta kernel is built for
//! `wasm32-unknown-unknown` and produces a `KernelDef` the existing
//! emitters (PTX / GCN / SPIR-V / MSL / WGSL) lower to backend ISAs.
//!
//! Today: just the parser entry point + side-table types. The actual
//! op-by-op lowering lands in subsequent commits.

#![allow(dead_code)]

use quanta_ir::{KernelDef, ScalarType};

/// Per-parameter metadata the kernel macro emits alongside the WASM,
/// telling the lowering pass which function parameters are buffer
/// pointers vs scalar push constants and which slot each maps to.
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

/// Side table the macro emits per kernel.
#[derive(Debug, Clone)]
pub struct SideTable {
    pub kernel_name: String,
    pub params: Vec<ParamSlot>,
    pub workgroup_size: [u32; 3],
}

/// Errors surfaced by the lowering pass.
#[derive(Debug)]
pub enum LoweringError {
    /// `wasmparser` rejected the module as malformed.
    Parse(String),
    /// The named function wasn't found in the WASM module.
    KernelNotFound(String),
    /// We encountered a WASM op the lowering pass doesn't yet support.
    /// `op` is the textual mnemonic; `at` is the byte offset for
    /// source-mapping.
    UnsupportedOp { op: String, at: usize },
    /// The function shape didn't match the side table.
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

/// Lower a WASM module + side table into a `KernelDef`.
///
/// Stubbed today; subsequent commits implement the op-by-op
/// translation.
pub fn lower(_wasm: &[u8], side_table: &SideTable) -> Result<KernelDef, LoweringError> {
    Err(LoweringError::UnsupportedOp {
        op: format!(
            "lowering not yet implemented (kernel `{}`)",
            side_table.kernel_name
        ),
        at: 0,
    })
}
