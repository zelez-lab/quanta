//! Parse a validated Rust function into KernelDef.
//!
//! Phase 2: full AST -> KernelOp walking via recursive emit_expr/emit_stmt.

mod expr;
mod stmt;

use quanta_ir::{BinOp, CmpOp, KernelDef, KernelOp, KernelParam, MathFn, Reg, ScalarType};
use std::collections::HashMap;
use syn::{BinOp as SynBinOp, Expr, FnArg, ItemFn, Pat, Type};

/// Parsed device function signature — enough to type-check calls.
#[derive(Clone)]
pub(crate) struct DeviceFnInfo {
    /// Parameter types (stored for future arity/type validation at call sites).
    #[allow(dead_code)]
    pub(crate) param_types: Vec<ScalarType>,
    pub(crate) return_type: ScalarType,
}

/// Emission context — tracks registers, variables, and parameters.
pub(crate) struct EmitCtx {
    pub(crate) ops: Vec<KernelOp>,
    pub(crate) next_reg: u32,
    /// Variable name -> (register, type)
    pub(crate) vars: HashMap<String, (Reg, ScalarType)>,
    /// Parameter name -> (slot, kind, type)
    pub(crate) params: HashMap<String, ParamInfo>,
    /// Shared memory counter
    pub(crate) next_shared: u32,
    /// Shared variable name -> (shared_id, element_type)
    pub(crate) shared_vars: HashMap<String, (u32, ScalarType)>,
    /// Device functions defined inside the kernel body (inner `fn` items).
    /// Maps function name -> signature info for call-site type resolution.
    pub(crate) device_fns: HashMap<String, DeviceFnInfo>,
    /// Collected source text of device functions, for MSL/WGSL emitters.
    pub(crate) device_sources: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct ParamInfo {
    pub(crate) slot: u32,
    pub(crate) is_const: bool,
    pub(crate) scalar_type: ScalarType,
}

impl EmitCtx {
    fn new(params: &[KernelParam]) -> Self {
        let mut param_map = HashMap::new();
        for p in params {
            match p {
                KernelParam::FieldRead {
                    name,
                    slot,
                    scalar_type,
                } => {
                    param_map.insert(
                        name.clone(),
                        ParamInfo {
                            slot: *slot,
                            is_const: false,
                            scalar_type: *scalar_type,
                        },
                    );
                }
                KernelParam::FieldWrite {
                    name,
                    slot,
                    scalar_type,
                } => {
                    param_map.insert(
                        name.clone(),
                        ParamInfo {
                            slot: *slot,
                            is_const: false,
                            scalar_type: *scalar_type,
                        },
                    );
                }
                KernelParam::Constant {
                    name,
                    slot,
                    scalar_type,
                } => {
                    param_map.insert(
                        name.clone(),
                        ParamInfo {
                            slot: *slot,
                            is_const: true,
                            scalar_type: *scalar_type,
                        },
                    );
                }
                _ => {}
            }
        }
        Self {
            ops: Vec::new(),
            next_reg: 0,
            vars: HashMap::new(),
            params: param_map,
            next_shared: 0,
            shared_vars: HashMap::new(),
            device_fns: HashMap::new(),
            device_sources: Vec::new(),
        }
    }

    pub(crate) fn alloc_reg(&mut self) -> Reg {
        let r = Reg(self.next_reg);
        self.next_reg += 1;
        r
    }

    /// Create a child context for loop/branch bodies that shares variables by reference.
    /// After emitting the body, call `merge_child` to propagate register count and var updates.
    pub(crate) fn child(&self) -> Self {
        Self {
            ops: Vec::new(),
            next_reg: self.next_reg,
            vars: self.vars.clone(),
            params: self.params.clone(),
            next_shared: self.next_shared,
            shared_vars: self.shared_vars.clone(),
            device_fns: self.device_fns.clone(),
            device_sources: Vec::new(), // collected at top level only
        }
    }

    /// Merge child context back: take its ops, update next_reg, propagate var remappings.
    pub(crate) fn merge_child(&mut self, child: Self) -> Vec<KernelOp> {
        self.next_reg = child.next_reg;
        // Propagate variable reassignments from child back to parent
        for (name, (reg, ty)) in &child.vars {
            if let Some((parent_reg, _)) = self.vars.get(name)
                && reg != parent_reg
            {
                // Variable was reassigned inside child scope — update parent
                self.vars.insert(name.clone(), (*reg, *ty));
            }
        }
        child.ops
    }
}

/// Parse a Rust function into KernelDef with populated body ops.
pub fn parse_kernel(func: &ItemFn) -> Result<KernelDef, syn::Error> {
    let name = func.sig.ident.to_string();
    let mut params = Vec::new();
    let mut slot = 0u32;

    for arg in &func.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            let param_name = match pat_type.pat.as_ref() {
                Pat::Ident(ident) => ident.ident.to_string(),
                _ => format!("param_{}", slot),
            };
            let param = parse_param_type(&param_name, &pat_type.ty, slot)?;
            params.push(param);
            slot += 1;
        }
    }

    let mut ctx = EmitCtx::new(&params);

    for s in &func.block.stmts {
        stmt::emit_stmt(s, &mut ctx)?;
    }

    Ok(KernelDef {
        name,
        params,
        body: ctx.ops,
        body_source: None,
        next_reg: ctx.next_reg,
        opt_level: 3, // overridden by proc macro attribute
        device_sources: ctx.device_sources,
    })
}

// ============================================================================
// Helpers
// ============================================================================

pub(crate) fn expr_to_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(path) => path.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

pub(crate) fn syn_binop_to_ir(op: &SynBinOp) -> Result<BinOp, syn::Error> {
    match op {
        SynBinOp::Add(_) => Ok(BinOp::Add),
        SynBinOp::Sub(_) => Ok(BinOp::Sub),
        SynBinOp::Mul(_) => Ok(BinOp::Mul),
        SynBinOp::Div(_) => Ok(BinOp::Div),
        SynBinOp::Rem(_) => Ok(BinOp::Rem),
        SynBinOp::BitAnd(_) => Ok(BinOp::BitAnd),
        SynBinOp::BitOr(_) => Ok(BinOp::BitOr),
        SynBinOp::BitXor(_) => Ok(BinOp::BitXor),
        SynBinOp::Shl(_) => Ok(BinOp::Shl),
        SynBinOp::Shr(_) => Ok(BinOp::Shr),
        _ => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "unsupported binary operator",
        )),
    }
}

pub(crate) fn syn_binop_to_cmp(op: &SynBinOp) -> Option<CmpOp> {
    match op {
        SynBinOp::Eq(_) => Some(CmpOp::Eq),
        SynBinOp::Ne(_) => Some(CmpOp::Ne),
        SynBinOp::Lt(_) => Some(CmpOp::Lt),
        SynBinOp::Le(_) => Some(CmpOp::Le),
        SynBinOp::Gt(_) => Some(CmpOp::Gt),
        SynBinOp::Ge(_) => Some(CmpOp::Ge),
        _ => None,
    }
}

pub(crate) fn is_assign_op(op: &SynBinOp) -> bool {
    matches!(
        op,
        SynBinOp::AddAssign(_)
            | SynBinOp::SubAssign(_)
            | SynBinOp::MulAssign(_)
            | SynBinOp::DivAssign(_)
            | SynBinOp::RemAssign(_)
    )
}

pub(crate) fn assign_op_to_binop(op: &SynBinOp) -> Result<BinOp, syn::Error> {
    match op {
        SynBinOp::AddAssign(_) => Ok(BinOp::Add),
        SynBinOp::SubAssign(_) => Ok(BinOp::Sub),
        SynBinOp::MulAssign(_) => Ok(BinOp::Mul),
        SynBinOp::DivAssign(_) => Ok(BinOp::Div),
        SynBinOp::RemAssign(_) => Ok(BinOp::Rem),
        _ => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "unsupported compound assignment",
        )),
    }
}

pub(crate) fn name_to_math_fn(name: &str) -> Option<MathFn> {
    match name {
        "sin" => Some(MathFn::Sin),
        "cos" => Some(MathFn::Cos),
        "tan" => Some(MathFn::Tan),
        "asin" => Some(MathFn::Asin),
        "acos" => Some(MathFn::Acos),
        "atan" => Some(MathFn::Atan),
        "atan2" => Some(MathFn::Atan2),
        "sqrt" => Some(MathFn::Sqrt),
        "rsqrt" => Some(MathFn::Rsqrt),
        "exp" => Some(MathFn::Exp),
        "exp2" => Some(MathFn::Exp2),
        "log" => Some(MathFn::Log),
        "log2" => Some(MathFn::Log2),
        "pow" => Some(MathFn::Pow),
        "abs" => Some(MathFn::Abs),
        "min" => Some(MathFn::Min),
        "max" => Some(MathFn::Max),
        "clamp" => Some(MathFn::Clamp),
        "floor" => Some(MathFn::Floor),
        "ceil" => Some(MathFn::Ceil),
        "round" => Some(MathFn::Round),
        "fma" => Some(MathFn::Fma),
        _ => None,
    }
}

// ============================================================================
// Parameter parsing
// ============================================================================

fn parse_param_type(name: &str, ty: &Type, slot: u32) -> Result<KernelParam, syn::Error> {
    match ty {
        Type::Reference(ref_ty) => {
            let is_mut = ref_ty.mutability.is_some();
            match ref_ty.elem.as_ref() {
                Type::Slice(slice) => {
                    let scalar = scalar_type_from_type(&slice.elem)?;
                    if is_mut {
                        Ok(KernelParam::FieldWrite {
                            name: name.to_string(),
                            slot,
                            scalar_type: scalar,
                        })
                    } else {
                        Ok(KernelParam::FieldRead {
                            name: name.to_string(),
                            slot,
                            scalar_type: scalar,
                        })
                    }
                }
                _ => Err(syn::Error::new_spanned(
                    ty,
                    "reference parameter must be &[T] or &mut [T]",
                )),
            }
        }
        Type::Path(path) => {
            let scalar = scalar_type_from_path(path)?;
            Ok(KernelParam::Constant {
                name: name.to_string(),
                slot,
                scalar_type: scalar,
            })
        }
        _ => Err(syn::Error::new_spanned(ty, "unsupported parameter type")),
    }
}

fn scalar_type_from_type(ty: &Type) -> Result<ScalarType, syn::Error> {
    match ty {
        Type::Path(path) => scalar_type_from_path(path),
        _ => Err(syn::Error::new_spanned(ty, "expected a scalar type")),
    }
}

pub(crate) fn scalar_type_from_path(path: &syn::TypePath) -> Result<ScalarType, syn::Error> {
    let ident = path
        .path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new_spanned(path, "empty type path"))?;

    match ident.ident.to_string().as_str() {
        "f16" => Ok(ScalarType::F16),
        "f32" => Ok(ScalarType::F32),
        "f64" => Ok(ScalarType::F64),
        "u8" => Ok(ScalarType::U8),
        "u16" => Ok(ScalarType::U16),
        "u32" => Ok(ScalarType::U32),
        "u64" => Ok(ScalarType::U64),
        "i8" => Ok(ScalarType::I8),
        "i16" => Ok(ScalarType::I16),
        "i32" => Ok(ScalarType::I32),
        "i64" => Ok(ScalarType::I64),
        "bool" => Ok(ScalarType::Bool),
        other => Err(syn::Error::new_spanned(
            &ident.ident,
            format!("unsupported GPU type: {}", other),
        )),
    }
}
