//! Kernel signature analysis — struct-ref parameter detection and
//! field-access scanning.
//!
//! These passes inspect `&MyStruct` parameters and the body's
//! `p.field` / `p.field[idx]` patterns to determine which struct
//! fields are read, written, and whether each is a buffer or a
//! scalar push constant. The output drives slot assignment for the
//! flattened kernel signature.
//!
//! This module deliberately operates only on the syn AST of the
//! function *signature* and the *shape* of body expressions — it
//! does NOT translate kernel bodies to KernelOps. That distinction
//! matters for the WASM-route cutover (step 058 + 059): the body
//! translator (`crate::parse`) goes away when rustc → wasm32 →
//! KernelOps replaces it, but the signature analysis here survives
//! unchanged. The cutover will call these functions, then build a
//! `quanta_wasm_lowering::SideTable` from the results before invoking
//! `lower_module`.

use std::collections::HashMap;

use syn::{BinOp as SynBinOp, Expr, FnArg, ItemFn, Pat, Stmt, Type};

/// A discovered field access on a struct-ref kernel parameter.
///
/// Collected by scanning the kernel body for patterns like:
/// - `p.pos[idx]`  — indexed buffer access (Vec<T> field)
/// - `p.count`     — scalar access (push constant)
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
    let path = match ref_ty.elem.as_ref() {
        Type::Path(p) => p,
        _ => return None,
    };
    let seg = path.path.segments.last()?;
    let type_name = seg.ident.to_string();

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
/// - `p.field[idx]` patterns — buffer fields (indexed reads/writes)
/// - `p.field` patterns — scalar fields (push constants)
///
/// Classifies each field as read, written, or both based on usage context.
pub(crate) fn scan_struct_field_accesses(
    func: &ItemFn,
    param_name: &str,
) -> Vec<StructFieldAccess> {
    // Map: name -> (is_indexed, is_read, is_written)
    let mut fields: HashMap<String, (bool, bool, bool)> = HashMap::new();

    for stmt in &func.block.stmts {
        scan_stmt_for_field_accesses(stmt, param_name, &mut fields);
    }

    // Order: buffers first, then scalars; sort by name for determinism.
    let mut buffer_fields: Vec<_> = fields
        .iter()
        .filter(|(_, (indexed, _, _))| *indexed)
        .collect();
    let mut scalar_fields: Vec<_> = fields
        .iter()
        .filter(|(_, (indexed, _, _))| !*indexed)
        .collect();
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
            scalar_type_name: String::new(),
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
            scalar_type_name: String::new(),
        });
        slot += 1;
    }

    result
}

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

/// `in_store_target` is true when this expression is on the left side of an assignment.
fn scan_expr_for_field_accesses(
    expr: &Expr,
    param_name: &str,
    fields: &mut HashMap<String, (bool, bool, bool)>,
    in_store_target: bool,
) {
    match expr {
        // p.field[idx] — indexed access: the Index wraps a Field expression
        Expr::Index(index) => {
            if let Some(field_name) = extract_struct_field_name(&index.expr, param_name) {
                let entry = fields.entry(field_name).or_insert((true, false, false));
                entry.0 = true;
                if in_store_target {
                    entry.2 = true;
                } else {
                    entry.1 = true;
                }
            }
            scan_expr_for_field_accesses(&index.index, param_name, fields, false);
            if extract_struct_field_name(&index.expr, param_name).is_none() {
                scan_expr_for_field_accesses(&index.expr, param_name, fields, false);
            }
        }
        // p.field — direct scalar access (push constant)
        Expr::Field(field_expr) => {
            if let Some(field_name) = extract_struct_field_name(expr, param_name) {
                let entry = fields.entry(field_name).or_insert((false, false, false));
                if in_store_target {
                    entry.2 = true;
                } else {
                    entry.1 = true;
                }
            } else {
                scan_expr_for_field_accesses(&field_expr.base, param_name, fields, false);
            }
        }
        Expr::Assign(assign) => {
            scan_expr_for_field_accesses(&assign.left, param_name, fields, true);
            scan_expr_for_field_accesses(&assign.right, param_name, fields, false);
        }
        Expr::Binary(bin) => {
            if is_assign_op(&bin.op) {
                scan_expr_for_field_accesses(&bin.left, param_name, fields, true);
                scan_expr_for_field_accesses(&bin.right, param_name, fields, false);
            } else {
                scan_expr_for_field_accesses(&bin.left, param_name, fields, false);
                scan_expr_for_field_accesses(&bin.right, param_name, fields, false);
            }
        }
        Expr::Block(block) => {
            for stmt in &block.block.stmts {
                scan_stmt_for_field_accesses(stmt, param_name, fields);
            }
        }
        Expr::If(if_expr) => {
            scan_expr_for_field_accesses(&if_expr.cond, param_name, fields, false);
            for stmt in &if_expr.then_branch.stmts {
                scan_stmt_for_field_accesses(stmt, param_name, fields);
            }
            if let Some((_, else_expr)) = &if_expr.else_branch {
                scan_expr_for_field_accesses(else_expr, param_name, fields, false);
            }
        }
        Expr::ForLoop(for_loop) => {
            scan_expr_for_field_accesses(&for_loop.expr, param_name, fields, false);
            for stmt in &for_loop.body.stmts {
                scan_stmt_for_field_accesses(stmt, param_name, fields);
            }
        }
        Expr::While(while_loop) => {
            scan_expr_for_field_accesses(&while_loop.cond, param_name, fields, false);
            for stmt in &while_loop.body.stmts {
                scan_stmt_for_field_accesses(stmt, param_name, fields);
            }
        }
        Expr::Unary(unary) => {
            scan_expr_for_field_accesses(&unary.expr, param_name, fields, false);
        }
        Expr::Call(call) => {
            for arg in &call.args {
                // Atomic targets: `atomic_add(&mut p.field[i], val)`
                if let Expr::Reference(ref_expr) = arg
                    && ref_expr.mutability.is_some()
                {
                    scan_expr_for_field_accesses(&ref_expr.expr, param_name, fields, true);
                    scan_expr_for_field_accesses(&ref_expr.expr, param_name, fields, false);
                    continue;
                }
                scan_expr_for_field_accesses(arg, param_name, fields, false);
            }
        }
        Expr::MethodCall(mc) => {
            scan_expr_for_field_accesses(&mc.receiver, param_name, fields, false);
            for arg in &mc.args {
                scan_expr_for_field_accesses(arg, param_name, fields, false);
            }
        }
        Expr::Paren(paren) => {
            scan_expr_for_field_accesses(&paren.expr, param_name, fields, in_store_target);
        }
        Expr::Cast(cast) => {
            scan_expr_for_field_accesses(&cast.expr, param_name, fields, false);
        }
        Expr::Reference(ref_expr) => {
            scan_expr_for_field_accesses(&ref_expr.expr, param_name, fields, in_store_target);
        }
        Expr::Range(range) => {
            if let Some(start) = &range.start {
                scan_expr_for_field_accesses(start, param_name, fields, false);
            }
            if let Some(end) = &range.end {
                scan_expr_for_field_accesses(end, param_name, fields, false);
            }
        }
        Expr::Return(ret) => {
            if let Some(inner) = &ret.expr {
                scan_expr_for_field_accesses(inner, param_name, fields, false);
            }
        }
        // Path, Lit, Break, Continue — leaf nodes, no field accesses.
        _ => {}
    }
}

/// Extract the field name from a `p.field_name` expression, if it matches
/// the given struct parameter name.
fn extract_struct_field_name(expr: &Expr, param_name: &str) -> Option<String> {
    if let Expr::Field(field_expr) = expr
        && let Expr::Path(path) = field_expr.base.as_ref()
        && let Some(seg) = path.path.segments.last()
        && seg.ident == param_name
        && let syn::Member::Named(ident) = &field_expr.member
    {
        return Some(ident.to_string());
    }
    None
}

/// Private duplicate of the `parse::is_assign_op` helper. Kept here
/// so this module is self-contained and survives the eventual
/// deletion of `parse.rs` during the WASM-route cutover.
fn is_assign_op(op: &SynBinOp) -> bool {
    matches!(
        op,
        SynBinOp::AddAssign(_)
            | SynBinOp::SubAssign(_)
            | SynBinOp::MulAssign(_)
            | SynBinOp::DivAssign(_)
            | SynBinOp::RemAssign(_)
    )
}
