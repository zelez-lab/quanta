#![allow(clippy::collapsible_if, dead_code)]
//! Parse a validated Rust function into KernelDef.
//!
//! Phase 2: full AST -> KernelOp walking via recursive emit_expr/emit_stmt.
//!
//! Supports two parameter styles:
//! - **Flat params:** `fn kernel(a: &[f32], b: &mut [f32], n: u32)` -- one KernelParam per arg
//! - **Struct-ref param:** `fn kernel(p: &MyStruct)` -- field accesses in the body are discovered
//!   via body scanning, then flattened into KernelParams

pub(crate) mod expr;
mod stmt;

use quanta_ir::{
    BinOp, CmpOp, DeviceFnDef, KernelDef, KernelOp, KernelParam, MathFn, Reg, ScalarType,
};
use std::collections::HashMap;
use syn::{BinOp as SynBinOp, Expr, FnArg, ItemFn, Pat, Stmt, Type};

// ============================================================================
// Struct-ref parameter support
// ============================================================================

/// A discovered field access on a struct-ref kernel parameter.
///
/// Collected by scanning the kernel body for patterns like:
/// - `p.pos[idx]`  -- indexed buffer access (Vec<T> field)
/// - `p.count`     -- scalar access (push constant)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct StructFieldAccess {
    /// Field name (e.g., "pos", "vel", "count").
    pub name: String,
    /// Slot index in the kernel param list.
    pub slot: usize,
    /// Whether the field is accessed with indexing (buffer) or not (scalar/push constant).
    pub is_indexed: bool,
    /// Whether the field is read by the kernel (load).
    pub is_read: bool,
    /// Whether the field is written by the kernel (store).
    pub is_written: bool,
    /// Scalar type name for code generation (e.g., "f32", "u32").
    /// For indexed fields, this is the element type of the Vec.
    /// For scalar fields, this is the scalar type itself.
    pub scalar_type_name: String,
}

/// Information about a detected struct-ref parameter.
pub(crate) struct StructRefParam {
    /// The parameter name (e.g., "p").
    pub param_name: String,
    /// The type name (e.g., "Particles").
    pub type_name: String,
    /// The full type path tokens.
    pub type_tokens: proc_macro2::TokenStream,
}

/// Detect if a function has a struct-ref parameter.
///
/// A struct-ref parameter is `p: &SomeStruct` where SomeStruct is NOT a slice,
/// NOT a texture, and NOT a primitive scalar type.
pub(crate) fn detect_struct_ref_param(func: &ItemFn) -> Option<StructRefParam> {
    // Only detect when there is exactly one parameter and it's a reference
    // to a named type (not a slice, not a primitive).
    if func.sig.inputs.len() != 1 {
        return None;
    }
    let arg = func.sig.inputs.first()?;
    let pat_type = match arg {
        FnArg::Typed(pt) => pt,
        _ => return None,
    };
    let param_name = match pat_type.pat.as_ref() {
        Pat::Ident(ident) => ident.ident.to_string(),
        _ => return None,
    };
    let ref_ty = match pat_type.ty.as_ref() {
        Type::Reference(r) => r,
        _ => return None,
    };
    // The inner type must be a path (not a slice)
    let path = match ref_ty.elem.as_ref() {
        Type::Path(p) => p,
        _ => return None,
    };
    let seg = path.path.segments.last()?;
    let type_name = seg.ident.to_string();

    // Exclude known primitive/GPU types -- these are handled as flat params
    if is_scalar_type_name(&type_name) || type_name == "Texture2D" || type_name == "Texture3D" {
        return None;
    }

    let type_tokens = quote::quote! { #path };

    Some(StructRefParam {
        param_name,
        type_name,
        type_tokens,
    })
}

fn is_scalar_type_name(name: &str) -> bool {
    matches!(
        name,
        "f16"
            | "f32"
            | "f64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "bool"
    )
}

/// Scan a function body to discover all field accesses on a struct-ref parameter.
///
/// Walks the AST looking for:
/// - `p.field[idx]` patterns -- buffer fields (indexed reads/writes)
/// - `p.field` patterns -- scalar fields (push constants)
///
/// Classifies each field as read, written, or both based on usage context.
pub(crate) fn scan_struct_field_accesses(
    func: &ItemFn,
    param_name: &str,
) -> Vec<StructFieldAccess> {
    let mut fields: HashMap<String, (bool, bool, bool)> = HashMap::new(); // name -> (is_indexed, is_read, is_written)

    for stmt in &func.block.stmts {
        scan_stmt_for_field_accesses(stmt, param_name, &mut fields);
    }

    // Convert to ordered list, assigning slots: buffers first, then scalars
    let mut buffer_fields: Vec<_> = fields
        .iter()
        .filter(|(_, (indexed, _, _))| *indexed)
        .collect();
    let mut scalar_fields: Vec<_> = fields
        .iter()
        .filter(|(_, (indexed, _, _))| !*indexed)
        .collect();

    // Sort by name for deterministic slot assignment
    buffer_fields.sort_by_key(|(name, _)| (*name).clone());
    scalar_fields.sort_by_key(|(name, _)| (*name).clone());

    let mut result = Vec::new();
    let mut slot = 0;

    for (name, (is_indexed, is_read, is_written)) in &buffer_fields {
        result.push(StructFieldAccess {
            name: (*name).clone(),
            slot,
            is_indexed: *is_indexed,
            is_read: *is_read,
            is_written: *is_written,
            scalar_type_name: String::new(), // filled in during parse
        });
        slot += 1;
    }
    for (name, (is_indexed, is_read, is_written)) in &scalar_fields {
        result.push(StructFieldAccess {
            name: (*name).clone(),
            slot,
            is_indexed: *is_indexed,
            is_read: *is_read,
            is_written: *is_written,
            scalar_type_name: String::new(), // filled in during parse
        });
        slot += 1;
    }

    result
}

/// Recursively scan a statement for struct field accesses.
fn scan_stmt_for_field_accesses(
    stmt: &Stmt,
    param_name: &str,
    fields: &mut HashMap<String, (bool, bool, bool)>,
) {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                scan_expr_for_field_accesses(&init.expr, param_name, fields, false);
            }
        }
        Stmt::Expr(expr, _) => {
            scan_expr_for_field_accesses(expr, param_name, fields, false);
        }
        _ => {}
    }
}

/// Recursively scan an expression for struct field accesses.
///
/// `in_store_target` is true when this expression is on the left side of an assignment.
fn scan_expr_for_field_accesses(
    expr: &Expr,
    param_name: &str,
    fields: &mut HashMap<String, (bool, bool, bool)>,
    in_store_target: bool,
) {
    match expr {
        // p.field[idx] -- indexed access: the Index wraps a Field expression
        Expr::Index(index) => {
            if let Some(field_name) = extract_struct_field_name(&index.expr, param_name) {
                let entry = fields.entry(field_name).or_insert((true, false, false));
                entry.0 = true; // is_indexed
                if in_store_target {
                    entry.2 = true; // is_written
                } else {
                    entry.1 = true; // is_read
                }
            }
            // Also scan the index expression for reads
            scan_expr_for_field_accesses(&index.index, param_name, fields, false);
            // And the indexed expression itself if it's not a struct field
            if extract_struct_field_name(&index.expr, param_name).is_none() {
                scan_expr_for_field_accesses(&index.expr, param_name, fields, false);
            }
        }
        // p.field -- direct scalar access (push constant)
        Expr::Field(field_expr) => {
            if let Some(field_name) = extract_struct_field_name(expr, param_name) {
                let entry = fields.entry(field_name).or_insert((false, false, false));
                // Not indexed -> scalar
                if in_store_target {
                    entry.2 = true;
                } else {
                    entry.1 = true;
                }
            } else {
                scan_expr_for_field_accesses(&field_expr.base, param_name, fields, false);
            }
        }
        // Assignment: left = right
        Expr::Assign(assign) => {
            scan_expr_for_field_accesses(&assign.left, param_name, fields, true);
            scan_expr_for_field_accesses(&assign.right, param_name, fields, false);
        }
        // Binary: a op b, including compound assignments (a += b)
        Expr::Binary(bin) => {
            if is_assign_op(&bin.op) {
                scan_expr_for_field_accesses(&bin.left, param_name, fields, true);
                scan_expr_for_field_accesses(&bin.right, param_name, fields, false);
            } else {
                scan_expr_for_field_accesses(&bin.left, param_name, fields, false);
                scan_expr_for_field_accesses(&bin.right, param_name, fields, false);
            }
        }
        // Block
        Expr::Block(block) => {
            for stmt in &block.block.stmts {
                scan_stmt_for_field_accesses(stmt, param_name, fields);
            }
        }
        // If expression
        Expr::If(if_expr) => {
            scan_expr_for_field_accesses(&if_expr.cond, param_name, fields, false);
            for stmt in &if_expr.then_branch.stmts {
                scan_stmt_for_field_accesses(stmt, param_name, fields);
            }
            if let Some((_, else_expr)) = &if_expr.else_branch {
                scan_expr_for_field_accesses(else_expr, param_name, fields, false);
            }
        }
        // For loop
        Expr::ForLoop(for_loop) => {
            scan_expr_for_field_accesses(&for_loop.expr, param_name, fields, false);
            for stmt in &for_loop.body.stmts {
                scan_stmt_for_field_accesses(stmt, param_name, fields);
            }
        }
        // While loop
        Expr::While(while_loop) => {
            scan_expr_for_field_accesses(&while_loop.cond, param_name, fields, false);
            for stmt in &while_loop.body.stmts {
                scan_stmt_for_field_accesses(stmt, param_name, fields);
            }
        }
        // Unary
        Expr::Unary(unary) => {
            scan_expr_for_field_accesses(&unary.expr, param_name, fields, false);
        }
        // Call
        Expr::Call(call) => {
            for arg in &call.args {
                // Check for atomic targets: atomic_add(&mut p.field[i], val)
                if let Expr::Reference(ref_expr) = arg {
                    if ref_expr.mutability.is_some() {
                        scan_expr_for_field_accesses(&ref_expr.expr, param_name, fields, true);
                        scan_expr_for_field_accesses(&ref_expr.expr, param_name, fields, false);
                        continue;
                    }
                }
                scan_expr_for_field_accesses(arg, param_name, fields, false);
            }
        }
        // Method call
        Expr::MethodCall(mc) => {
            scan_expr_for_field_accesses(&mc.receiver, param_name, fields, false);
            for arg in &mc.args {
                scan_expr_for_field_accesses(arg, param_name, fields, false);
            }
        }
        // Paren
        Expr::Paren(paren) => {
            scan_expr_for_field_accesses(&paren.expr, param_name, fields, in_store_target);
        }
        // Cast
        Expr::Cast(cast) => {
            scan_expr_for_field_accesses(&cast.expr, param_name, fields, false);
        }
        // Reference (&x, &mut x)
        Expr::Reference(ref_expr) => {
            scan_expr_for_field_accesses(&ref_expr.expr, param_name, fields, in_store_target);
        }
        // Range (start..end)
        Expr::Range(range) => {
            if let Some(start) = &range.start {
                scan_expr_for_field_accesses(start, param_name, fields, false);
            }
            if let Some(end) = &range.end {
                scan_expr_for_field_accesses(end, param_name, fields, false);
            }
        }
        // Return
        Expr::Return(ret) => {
            if let Some(inner) = &ret.expr {
                scan_expr_for_field_accesses(inner, param_name, fields, false);
            }
        }
        // Path, Lit, Break, Continue -- leaf nodes, no field accesses
        _ => {}
    }
}

/// Infer the scalar type of a struct push-constant field by walking
/// the kernel body looking for usage-context hints. Returns `None` if
/// no hint can be derived; caller should fall back to the historical
/// default (`ScalarType::U32`).
///
/// Why this exists: the body parser refines *buffer-field* scalar
/// types at write sites (parse/stmt.rs:561 — Path A, commit 1cedee5),
/// but `Constant` push-const scalars never get refined by body parse.
/// Without this pre-pass, an `f32` push-const is loaded with kernel-
/// side type `uint` and the dispatch path's verbatim f32 byte upload
/// gets read as a u32 bit pattern (e.g., `0.5_f32` → `1056964608`).
///
/// The hints we recognise (in priority order):
/// 1. **Type annotation**: `let x: f32 = p.field;`
/// 2. **Cast on the other side of a binop**: `(i as f32) * p.field`
/// 3. **Literal on the other side of a binop**: `p.field * 2.0_f32`
/// 4. **`as` cast applied to the field directly**: `(p.field as f32)`
///    — though this is unusual; the result would already be f32.
///
/// Heuristic only — doesn't promise to catch every case. When a hint
/// can't be derived, the default is preserved and the user can either
/// adjust their kernel or use the manual API.
pub(crate) fn infer_const_scalar_type(
    func: &ItemFn,
    param_name: &str,
    field_name: &str,
) -> Option<ScalarType> {
    let mut hint: Option<ScalarType> = None;
    walk_block_for_const_hint(&func.block, param_name, field_name, &mut hint);
    hint
}

fn walk_block_for_const_hint(
    block: &syn::Block,
    param_name: &str,
    field_name: &str,
    hint: &mut Option<ScalarType>,
) {
    for stmt in &block.stmts {
        walk_stmt_for_const_hint(stmt, param_name, field_name, hint);
        if hint.is_some() {
            return;
        }
    }
}

fn walk_stmt_for_const_hint(
    stmt: &Stmt,
    param_name: &str,
    field_name: &str,
    hint: &mut Option<ScalarType>,
) {
    match stmt {
        // `let x: T = p.field` — direct type annotation.
        Stmt::Local(local) => {
            if let syn::Pat::Type(pat_ty) = &local.pat {
                if let Some(init) = &local.init {
                    if expr_is_field(&init.expr, param_name, field_name) {
                        if let Some(ty) = ty_to_scalar(&pat_ty.ty) {
                            *hint = Some(ty);
                            return;
                        }
                    }
                }
            }
            if let Some(init) = &local.init {
                walk_expr_for_const_hint(&init.expr, param_name, field_name, hint);
            }
        }
        Stmt::Expr(e, _) => walk_expr_for_const_hint(e, param_name, field_name, hint),
        Stmt::Macro(_) | Stmt::Item(_) => {}
    }
}

fn walk_expr_for_const_hint(
    expr: &Expr,
    param_name: &str,
    field_name: &str,
    hint: &mut Option<ScalarType>,
) {
    if hint.is_some() {
        return;
    }
    match expr {
        Expr::Binary(b) => {
            // If one side is `p.field`, try to infer from the other.
            let left_is_field = expr_is_field(&b.left, param_name, field_name);
            let right_is_field = expr_is_field(&b.right, param_name, field_name);
            if left_is_field {
                if let Some(ty) = expr_to_scalar(&b.right) {
                    *hint = Some(ty);
                    return;
                }
            }
            if right_is_field {
                if let Some(ty) = expr_to_scalar(&b.left) {
                    *hint = Some(ty);
                    return;
                }
            }
            walk_expr_for_const_hint(&b.left, param_name, field_name, hint);
            if hint.is_none() {
                walk_expr_for_const_hint(&b.right, param_name, field_name, hint);
            }
        }
        Expr::Cast(c) => {
            // `(p.field as T)` — the cast target is the field's effective type
            // *after* the cast. The pre-cast type is whatever the field is.
            // We cannot recover that from the cast alone; skip this case.
            walk_expr_for_const_hint(&c.expr, param_name, field_name, hint);
        }
        Expr::Paren(p) => walk_expr_for_const_hint(&p.expr, param_name, field_name, hint),
        Expr::Group(g) => walk_expr_for_const_hint(&g.expr, param_name, field_name, hint),
        Expr::Block(b) => walk_block_for_const_hint(&b.block, param_name, field_name, hint),
        Expr::If(i) => {
            walk_expr_for_const_hint(&i.cond, param_name, field_name, hint);
            walk_block_for_const_hint(&i.then_branch, param_name, field_name, hint);
            if let Some((_, else_branch)) = &i.else_branch {
                walk_expr_for_const_hint(else_branch, param_name, field_name, hint);
            }
        }
        Expr::While(w) => {
            walk_expr_for_const_hint(&w.cond, param_name, field_name, hint);
            walk_block_for_const_hint(&w.body, param_name, field_name, hint);
        }
        Expr::ForLoop(f) => {
            walk_expr_for_const_hint(&f.expr, param_name, field_name, hint);
            walk_block_for_const_hint(&f.body, param_name, field_name, hint);
        }
        Expr::Assign(a) => {
            // `p.buffer[idx] = p.field * something` — recurse into RHS;
            // the LHS being a typed buffer slot is also a hint, but
            // we don't yet plumb that through.
            walk_expr_for_const_hint(&a.left, param_name, field_name, hint);
            if hint.is_none() {
                walk_expr_for_const_hint(&a.right, param_name, field_name, hint);
            }
        }
        Expr::Call(c) => {
            for arg in &c.args {
                walk_expr_for_const_hint(arg, param_name, field_name, hint);
                if hint.is_some() {
                    return;
                }
            }
        }
        Expr::MethodCall(m) => {
            walk_expr_for_const_hint(&m.receiver, param_name, field_name, hint);
            for arg in &m.args {
                walk_expr_for_const_hint(arg, param_name, field_name, hint);
                if hint.is_some() {
                    return;
                }
            }
        }
        Expr::Index(i) => {
            walk_expr_for_const_hint(&i.expr, param_name, field_name, hint);
            if hint.is_none() {
                walk_expr_for_const_hint(&i.index, param_name, field_name, hint);
            }
        }
        Expr::Unary(u) => walk_expr_for_const_hint(&u.expr, param_name, field_name, hint),
        Expr::Return(r) => {
            if let Some(e) = &r.expr {
                walk_expr_for_const_hint(e, param_name, field_name, hint);
            }
        }
        _ => {}
    }
}

/// Returns true if `expr` is exactly `param.field_name`, OR
/// `param.field_name[any_index]` (buffer indexed access).
/// Both forms produce a Load that needs the right scalar type.
fn expr_is_field(expr: &Expr, param_name: &str, field_name: &str) -> bool {
    // p.field — direct scalar access
    if let Expr::Field(f) = expr {
        if let Expr::Path(p) = f.base.as_ref() {
            if let Some(seg) = p.path.segments.last() {
                if seg.ident == param_name {
                    if let syn::Member::Named(ident) = &f.member {
                        return ident == field_name;
                    }
                }
            }
        }
    }
    // p.field[idx] — buffer indexed access
    if let Expr::Index(i) = expr {
        if let Expr::Field(f) = i.expr.as_ref() {
            if let Expr::Path(p) = f.base.as_ref() {
                if let Some(seg) = p.path.segments.last() {
                    if seg.ident == param_name {
                        if let syn::Member::Named(ident) = &f.member {
                            return ident == field_name;
                        }
                    }
                }
            }
        }
    }
    false
}

/// If `expr` carries a definite scalar type (cast or typed literal),
/// return it. Otherwise None.
fn expr_to_scalar(expr: &Expr) -> Option<ScalarType> {
    match expr {
        Expr::Cast(c) => ty_to_scalar(&c.ty),
        Expr::Lit(l) => lit_to_scalar(&l.lit),
        Expr::Paren(p) => expr_to_scalar(&p.expr),
        Expr::Group(g) => expr_to_scalar(&g.expr),
        Expr::Unary(u) => expr_to_scalar(&u.expr),
        _ => None,
    }
}

fn ty_to_scalar(ty: &Type) -> Option<ScalarType> {
    let path = match ty {
        Type::Path(p) => p,
        Type::Reference(r) => return ty_to_scalar(&r.elem),
        Type::Paren(p) => return ty_to_scalar(&p.elem),
        _ => return None,
    };
    let seg = path.path.segments.last()?;
    Some(match seg.ident.to_string().as_str() {
        "f16" => ScalarType::F16,
        "f32" => ScalarType::F32,
        "f64" => ScalarType::F64,
        "u8" => ScalarType::U8,
        "u16" => ScalarType::U16,
        "u32" => ScalarType::U32,
        "u64" => ScalarType::U64,
        "i8" => ScalarType::I8,
        "i16" => ScalarType::I16,
        "i32" => ScalarType::I32,
        "i64" => ScalarType::I64,
        "bool" => ScalarType::Bool,
        _ => return None,
    })
}

fn lit_to_scalar(lit: &syn::Lit) -> Option<ScalarType> {
    match lit {
        syn::Lit::Float(f) => match f.suffix() {
            "f16" => Some(ScalarType::F16),
            "f32" | "" => Some(ScalarType::F32),
            "f64" => Some(ScalarType::F64),
            _ => None,
        },
        syn::Lit::Int(i) => match i.suffix() {
            "u8" => Some(ScalarType::U8),
            "u16" => Some(ScalarType::U16),
            "u32" => Some(ScalarType::U32),
            "u64" => Some(ScalarType::U64),
            "i8" => Some(ScalarType::I8),
            "i16" => Some(ScalarType::I16),
            "i32" => Some(ScalarType::I32),
            "i64" => Some(ScalarType::I64),
            "" => None, // unsuffixed int literal — ambiguous
            _ => None,
        },
        syn::Lit::Bool(_) => Some(ScalarType::Bool),
        _ => None,
    }
}

/// Extract the field name from a `p.field_name` expression, if it matches
/// the given struct parameter name.
fn extract_struct_field_name(expr: &Expr, param_name: &str) -> Option<String> {
    if let Expr::Field(field_expr) = expr {
        // Check that the base is the parameter name
        if let Expr::Path(path) = field_expr.base.as_ref() {
            if let Some(seg) = path.path.segments.last() {
                if seg.ident == param_name {
                    // Extract the field name from the member
                    if let syn::Member::Named(ident) = &field_expr.member {
                        return Some(ident.to_string());
                    }
                }
            }
        }
    }
    None
}

// ============================================================================
// Core types
// ============================================================================

/// Parsed device function signature -- enough to type-check calls.
#[derive(Clone)]
pub(crate) struct DeviceFnInfo {
    /// Parameter types (stored for future arity/type validation at call sites).
    #[allow(dead_code)]
    pub(crate) param_types: Vec<ScalarType>,
    pub(crate) return_type: ScalarType,
}

/// Emission context -- tracks registers, variables, and parameters.
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
    /// Parsed device function definitions with KernelOp bodies.
    pub(crate) device_functions: Vec<DeviceFnDef>,
    /// When a struct-ref parameter is used, this is the parameter name (e.g., "p").
    /// Enables `p.field[idx]` and `p.field` resolution in expr.rs.
    pub(crate) struct_ref_param: Option<String>,
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
                KernelParam::Texture2DRead {
                    name,
                    slot,
                    scalar_type,
                }
                | KernelParam::Texture2DWrite {
                    name,
                    slot,
                    scalar_type,
                }
                | KernelParam::Texture3DRead {
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
            device_functions: Vec::new(),
            struct_ref_param: None,
        }
    }

    pub(crate) fn alloc_reg(&mut self) -> Reg {
        let r = Reg(self.next_reg);
        self.next_reg += 1;
        r
    }

    pub(crate) fn param_slot(&self, name: &str) -> Option<u32> {
        self.params.get(name).map(|p| p.slot)
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
            device_sources: Vec::new(),   // collected at top level only
            device_functions: Vec::new(), // collected at top level only
            struct_ref_param: self.struct_ref_param.clone(),
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
                // Variable was reassigned inside child scope -- update parent
                self.vars.insert(name.clone(), (*reg, *ty));
            }
        }
        // Propagate scalar_type refinements made inside the child
        // scope back to the parent — needed so retroactive type
        // patches done inside loop bodies (e.g. expr.rs's
        // retypecast_load_chain_to_int for read-only u32 buffers)
        // survive the loop scope. Without this, Path A's end-of-parse
        // projection sees the parent's stale (default F32) type and
        // the auto-dispatch type-probe rejects the user's `Vec<u32>`.
        for (name, child_info) in &child.params {
            if let Some(parent_info) = self.params.get_mut(name) {
                if parent_info.scalar_type != child_info.scalar_type {
                    parent_info.scalar_type = child_info.scalar_type;
                }
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
    let mut struct_ref_name: Option<String> = None;

    // Check for struct-ref parameter mode
    let struct_ref = detect_struct_ref_param(func);

    if let Some(ref sr) = struct_ref {
        // Struct-ref mode: scan body for field accesses, build params from those
        let field_accesses = scan_struct_field_accesses(func, &sr.param_name);
        struct_ref_name = Some(sr.param_name.clone());

        for access in &field_accesses {
            if access.is_indexed {
                // Buffer field: determine read/write from access patterns
                if access.is_written {
                    params.push(KernelParam::FieldWrite {
                        name: access.name.clone(),
                        slot,
                        scalar_type: ScalarType::F32, // default, refined by body parse
                    });
                } else {
                    params.push(KernelParam::FieldRead {
                        name: access.name.clone(),
                        slot,
                        scalar_type: ScalarType::F32,
                    });
                }
            } else {
                // Scalar field: push constant.
                //
                // The body parser only refines *buffer* field types
                // (via stmt::emit_stmt's index-store case at parse/
                // stmt.rs:561). Push-const scalar types stay at their
                // default unless we infer them from usage context up
                // front. Without this inference, a struct field like
                // `scale: f32` gets kernel-side type `uint`; the
                // dispatch path uploads f32 bytes verbatim, the
                // kernel reads them as u32, and arithmetic on the
                // resulting Reg uses the bit-pattern integer instead
                // of the float — producing nonsense output (e.g.
                // `0.5_f32` ↦ `1056964608`). See
                // research/dual_form_layer_norm_gpu_probe/README.md
                // for the diagnostic.
                let inferred = infer_const_scalar_type(func, &sr.param_name, &access.name)
                    .unwrap_or(ScalarType::U32);
                params.push(KernelParam::Constant {
                    name: access.name.clone(),
                    slot,
                    scalar_type: inferred,
                });
            }
            slot += 1;
        }
    } else {
        // Flat parameter mode: existing behavior
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
    }

    // Const generics: treated as push constant parameters
    for generic in &func.sig.generics.params {
        if let syn::GenericParam::Const(cp) = generic {
            let const_name = cp.ident.to_string();
            params.push(KernelParam::Constant {
                name: const_name,
                slot,
                scalar_type: ScalarType::U32,
            });
            slot += 1;
        }
    }

    let mut ctx = EmitCtx::new(&params);
    ctx.struct_ref_param = struct_ref_name;

    for s in &func.block.stmts {
        stmt::emit_stmt(s, &mut ctx)?;
    }

    // Path A (roadmap step 080): refine each param's scalar_type from the
    // body-parse context. The struct-ref pass initially pushes params with
    // placeholder types (F32 for buffers, U32 for scalars); the body
    // parser updates ctx.params.scalar_type as it observes actual writes
    // and reads. Project those refinements back into the final params
    // vec so the KernelDef carries the user's true element types.
    for p in params.iter_mut() {
        let (name, slot) = match p {
            KernelParam::FieldRead { name, slot, .. }
            | KernelParam::FieldWrite { name, slot, .. }
            | KernelParam::Constant { name, slot, .. }
            | KernelParam::Texture2DRead { name, slot, .. }
            | KernelParam::Texture2DWrite { name, slot, .. }
            | KernelParam::Texture3DRead { name, slot, .. } => (name.clone(), *slot),
        };
        if let Some(info) = ctx.params.get(&name) {
            if info.slot == slot {
                let new_ty = info.scalar_type;
                match p {
                    KernelParam::FieldRead { scalar_type, .. }
                    | KernelParam::FieldWrite { scalar_type, .. }
                    | KernelParam::Constant { scalar_type, .. }
                    | KernelParam::Texture2DRead { scalar_type, .. }
                    | KernelParam::Texture2DWrite { scalar_type, .. }
                    | KernelParam::Texture3DRead { scalar_type, .. } => {
                        *scalar_type = new_ty;
                    }
                }
            }
        }
    }

    Ok(KernelDef {
        name,
        params,
        body: ctx.ops,
        body_source: None,
        next_reg: ctx.next_reg,
        opt_level: 3, // overridden by proc macro attribute
        device_sources: ctx.device_sources,
        device_functions: ctx.device_functions,
        workgroup_size: [64, 1, 1], // overridden by proc macro attribute
        subgroup_size: None,        // overridden by proc macro attribute
        dynamic_shared_bytes: 0,    // set by dispatch API
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
                Type::Path(path) => {
                    // Check for Texture2D<T>
                    if let Some(seg) = path.path.segments.last() {
                        let ident = seg.ident.to_string();
                        if ident == "Texture2D" {
                            let scalar = extract_generic_scalar(seg)?;
                            return if is_mut {
                                Ok(KernelParam::Texture2DWrite {
                                    name: name.to_string(),
                                    slot,
                                    scalar_type: scalar,
                                })
                            } else {
                                Ok(KernelParam::Texture2DRead {
                                    name: name.to_string(),
                                    slot,
                                    scalar_type: scalar,
                                })
                            };
                        }
                    }
                    Err(syn::Error::new_spanned(
                        ty,
                        "reference parameter must be &[T], &Texture2D<T>, or scalar &T",
                    ))
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

fn extract_generic_scalar(seg: &syn::PathSegment) -> Result<ScalarType, syn::Error> {
    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
        && let Some(syn::GenericArgument::Type(Type::Path(p))) = args.args.first()
    {
        return scalar_type_from_path(p);
    }
    Ok(ScalarType::F32) // default to f32
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
