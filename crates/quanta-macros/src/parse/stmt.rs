//! Statement emission — emit_stmt, loops, assignments.

use quanta_ir::{ConstValue, KernelOp, Reg, ScalarType, UnaryOp};
use std::collections::HashMap;
use syn::{Expr, Pat, Stmt, Type};

use super::expr::{emit_expr, emit_expr_stmt};
use super::{EmitCtx, assign_op_to_binop, expr_to_name, scalar_type_from_path};

pub(crate) fn emit_stmt(stmt: &Stmt, ctx: &mut EmitCtx) -> Result<(), syn::Error> {
    match stmt {
        Stmt::Local(local) => emit_local(local, ctx),
        Stmt::Expr(expr, _semi) => {
            emit_expr_stmt(expr, ctx)?;
            Ok(())
        }
        _ => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "unsupported statement in GPU kernel",
        )),
    }
}

/// Check if a `let` statement has a `#[quanta::shared]` (or `#[shared]`) attribute.
fn has_shared_attr(local: &syn::Local) -> bool {
    local.attrs.iter().any(|attr| {
        let path = attr.path();
        let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        // Match both #[quanta::shared] and #[shared]
        match segments.as_slice() {
            [single] => single == "shared",
            [ns, name] => ns == "quanta" && name == "shared",
            _ => false,
        }
    })
}

/// Parse an array type `[ScalarType; count]` from a `syn::Type`.
fn parse_shared_array_type(ty: &Type) -> Result<(ScalarType, u32), syn::Error> {
    match ty {
        Type::Array(array) => {
            let elem_ty = match array.elem.as_ref() {
                Type::Path(path) => scalar_type_from_path(path)?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        &array.elem,
                        "shared memory element must be a scalar type",
                    ));
                }
            };
            let count = match &array.len {
                syn::Expr::Lit(lit) => match &lit.lit {
                    syn::Lit::Int(i) => i
                        .base10_parse::<u32>()
                        .map_err(|e| syn::Error::new_spanned(i, e))?,
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &array.len,
                            "shared memory size must be an integer literal",
                        ));
                    }
                },
                _ => {
                    return Err(syn::Error::new_spanned(
                        &array.len,
                        "shared memory size must be an integer literal",
                    ));
                }
            };
            Ok((elem_ty, count))
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            "shared memory must be declared as [Type; count]",
        )),
    }
}

/// Emit a let binding, handling simple idents, tuple patterns, and shared memory declarations.
fn emit_local(local: &syn::Local, ctx: &mut EmitCtx) -> Result<(), syn::Error> {
    // Check for #[quanta::shared] attribute
    if has_shared_attr(local) {
        return emit_shared_decl(local, ctx);
    }

    match &local.pat {
        Pat::Ident(ident) => {
            let var_name = ident.ident.to_string();
            if let Some(init) = &local.init {
                let (reg, ty) = emit_expr(&init.expr, ctx)?;
                ctx.vars.insert(var_name, (reg, ty));
            }
            Ok(())
        }
        // Tuple pattern: let (mut x, mut y) = (expr1, expr2)
        Pat::Tuple(tuple) => {
            if let Some(init) = &local.init {
                // The RHS must be a tuple expression
                if let Expr::Tuple(rhs_tuple) = init.expr.as_ref() {
                    if tuple.elems.len() != rhs_tuple.elems.len() {
                        return Err(syn::Error::new_spanned(
                            &local.pat,
                            "tuple pattern length mismatch",
                        ));
                    }
                    for (pat, expr) in tuple.elems.iter().zip(rhs_tuple.elems.iter()) {
                        let var_name = match pat {
                            Pat::Ident(ident) => ident.ident.to_string(),
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    pat,
                                    "unsupported pattern in tuple binding",
                                ));
                            }
                        };
                        let (reg, ty) = emit_expr(expr, ctx)?;
                        ctx.vars.insert(var_name, (reg, ty));
                    }
                    Ok(())
                } else {
                    Err(syn::Error::new_spanned(
                        &init.expr,
                        "tuple pattern requires tuple expression on RHS",
                    ))
                }
            } else {
                Err(syn::Error::new_spanned(
                    &local.pat,
                    "tuple binding requires initializer",
                ))
            }
        }
        _ => Err(syn::Error::new_spanned(
            &local.pat,
            "unsupported pattern in let binding",
        )),
    }
}

/// Emit a shared memory declaration: `#[quanta::shared] let local: [f32; 256];`
fn emit_shared_decl(local: &syn::Local, ctx: &mut EmitCtx) -> Result<(), syn::Error> {
    let var_name = match &local.pat {
        Pat::Ident(ident) => ident.ident.to_string(),
        Pat::Type(pat_type) => match pat_type.pat.as_ref() {
            Pat::Ident(ident) => ident.ident.to_string(),
            _ => {
                return Err(syn::Error::new_spanned(
                    &local.pat,
                    "shared memory variable must be a simple name",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &local.pat,
                "shared memory variable must be a simple name",
            ));
        }
    };

    // Extract the type annotation — either from Pat::Type or from Local::ty (if present)
    let ty_ref = match &local.pat {
        Pat::Type(pat_type) => Some(pat_type.ty.as_ref()),
        _ => None,
    };

    let ty = ty_ref.ok_or_else(|| {
        syn::Error::new_spanned(
            &local.pat,
            "shared memory must have a type annotation: #[quanta::shared] let name: [Type; count];",
        )
    })?;

    let (scalar_ty, count) = parse_shared_array_type(ty)?;
    let id = ctx.next_shared;
    ctx.next_shared += 1;

    ctx.ops.push(KernelOp::SharedDecl {
        id,
        ty: scalar_ty,
        count,
    });
    ctx.shared_vars.insert(var_name, (id, scalar_ty));

    Ok(())
}

pub(crate) fn emit_for_loop(
    for_loop: &syn::ExprForLoop,
    ctx: &mut EmitCtx,
) -> Result<(), syn::Error> {
    // for i in 0..N { body }
    // Accept simple idents AND underscore/wildcard patterns
    let iter_name = match &*for_loop.pat {
        Pat::Ident(ident) => ident.ident.to_string(),
        Pat::Wild(_) => "_".to_string(),
        _ => {
            return Err(syn::Error::new_spanned(
                &for_loop.pat,
                "for loop variable must be a simple name or _",
            ));
        }
    };

    // Parse range: 0..N
    let count_reg = match &*for_loop.expr {
        Expr::Range(range) => {
            if let Some(end) = &range.end {
                let (r, _) = emit_expr(end, ctx)?;
                r
            } else {
                return Err(syn::Error::new_spanned(
                    &for_loop.expr,
                    "for loop requires a bounded range (0..N)",
                ));
            }
        }
        _ => {
            return Err(syn::Error::new_spanned(
                &for_loop.expr,
                "for loop must use a range (0..N)",
            ));
        }
    };

    let iter_reg = ctx.alloc_reg();
    // Only register the iteration variable if it's not a wildcard
    if iter_name != "_" {
        ctx.vars.insert(iter_name, (iter_reg, ScalarType::U32));
    }

    // Snapshot variable registers before the loop body
    let vars_before: HashMap<String, (Reg, ScalarType)> = ctx.vars.clone();

    // Body
    let mut body_ctx = ctx.child();
    for stmt in &for_loop.body.stmts {
        emit_stmt(stmt, &mut body_ctx)?;
    }

    // Emit copies for loop-carried variables: copy new register back to original
    for (name, (orig_reg, ty)) in &vars_before {
        if let Some(&(new_reg, _)) = body_ctx.vars.get(name)
            && new_reg != *orig_reg
        {
            body_ctx.ops.push(KernelOp::Copy {
                dst: *orig_reg,
                src: new_reg,
                ty: *ty,
            });
            // Reset the child's var mapping to the original register
            // so that merge_child doesn't change the parent's mapping
            body_ctx.vars.insert(name.clone(), (*orig_reg, *ty));
        }
    }

    let body_ops = ctx.merge_child(body_ctx);

    ctx.ops.push(KernelOp::Loop {
        count: count_reg,
        iter_reg,
        body: body_ops,
    });
    Ok(())
}

pub(crate) fn emit_while_loop(
    while_loop: &syn::ExprWhile,
    ctx: &mut EmitCtx,
) -> Result<(), syn::Error> {
    // while cond { body } -> for (_w = 0; _w < 10000; _w++) { if !cond { break; } body; }
    // GPU kernels must be bounded, so we use a max iteration count as a safety limit.
    let max_iter = 10000u32;
    let max_reg = ctx.alloc_reg();
    ctx.ops.push(KernelOp::Const {
        dst: max_reg,
        value: ConstValue::U32(max_iter),
    });

    let iter_reg = ctx.alloc_reg();

    // Snapshot variable registers before the loop body
    let vars_before: HashMap<String, (Reg, ScalarType)> = ctx.vars.clone();

    // Build the body: first check condition, break if false, then run actual body
    let mut body_ctx = ctx.child();

    // Emit condition check
    let (cond_reg, _) = emit_expr(&while_loop.cond, &mut body_ctx)?;

    // if !cond { break; }
    let not_cond = body_ctx.alloc_reg();
    body_ctx.ops.push(KernelOp::UnaryOp {
        dst: not_cond,
        a: cond_reg,
        op: UnaryOp::LogicalNot,
        ty: ScalarType::Bool,
    });
    body_ctx.ops.push(KernelOp::Branch {
        cond: not_cond,
        then_ops: vec![KernelOp::Break],
        else_ops: vec![],
    });

    // Emit actual body
    for stmt in &while_loop.body.stmts {
        emit_stmt(stmt, &mut body_ctx)?;
    }

    // Emit copies for loop-carried variables: copy new register back to original
    for (name, (orig_reg, ty)) in &vars_before {
        if let Some(&(new_reg, _)) = body_ctx.vars.get(name)
            && new_reg != *orig_reg
        {
            body_ctx.ops.push(KernelOp::Copy {
                dst: *orig_reg,
                src: new_reg,
                ty: *ty,
            });
            body_ctx.vars.insert(name.clone(), (*orig_reg, *ty));
        }
    }

    let body_ops = ctx.merge_child(body_ctx);

    ctx.ops.push(KernelOp::Loop {
        count: max_reg,
        iter_reg,
        body: body_ops,
    });
    Ok(())
}

/// Store to a field[index], shared[index], or reassign a local variable.
pub(crate) fn emit_store_or_reassign(
    target: &Expr,
    src_reg: Reg,
    src_ty: ScalarType,
    ctx: &mut EmitCtx,
) -> Result<(), syn::Error> {
    match target {
        // field[index] = value  OR  shared[index] = value
        Expr::Index(index) => {
            let arr_name = expr_to_name(&index.expr).ok_or_else(|| {
                syn::Error::new_spanned(&index.expr, "store target must be a field name")
            })?;

            // Check shared variables first
            if let Some(&(shared_id, scalar_ty)) = ctx.shared_vars.get(&arr_name) {
                let (idx_reg, _) = emit_expr(&index.index, ctx)?;
                ctx.ops.push(KernelOp::SharedStore {
                    id: shared_id,
                    index: idx_reg,
                    src: src_reg,
                    ty: scalar_ty,
                });
                return Ok(());
            }

            let info = ctx
                .params
                .get(&arr_name)
                .ok_or_else(|| {
                    syn::Error::new_spanned(&index.expr, format!("unknown field: {}", arr_name))
                })?
                .clone();
            let (idx_reg, _) = emit_expr(&index.index, ctx)?;
            ctx.ops.push(KernelOp::Store {
                field: info.slot,
                index: idx_reg,
                src: src_reg,
                ty: info.scalar_type,
            });
            Ok(())
        }
        // x = value (local variable reassignment)
        Expr::Path(path) => {
            let name = path
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            match ctx.vars.entry(name) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    e.insert((src_reg, src_ty));
                    Ok(())
                }
                std::collections::hash_map::Entry::Vacant(e) => Err(syn::Error::new_spanned(
                    target,
                    format!("cannot assign to undefined variable: {}", e.key()),
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            target,
            "assignment target must be field[index] or a local variable",
        )),
    }
}

/// Handle compound assignment: a[i] += expr  OR  x += expr
pub(crate) fn emit_compound_assign(
    bin: &syn::ExprBinary,
    ctx: &mut EmitCtx,
) -> Result<(), syn::Error> {
    let op = assign_op_to_binop(&bin.op)?;

    match &*bin.left {
        // Compound assignment on a local variable: x += expr
        Expr::Path(path) => {
            let name = path
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if let Some(&(left_reg, ty)) = ctx.vars.get(&name) {
                let (right_reg, _) = emit_expr(&bin.right, ctx)?;
                let dst = ctx.alloc_reg();
                ctx.ops.push(KernelOp::BinOp {
                    dst,
                    a: left_reg,
                    b: right_reg,
                    op,
                    ty,
                });
                ctx.vars.insert(name, (dst, ty));
                Ok(())
            } else {
                Err(syn::Error::new_spanned(
                    &bin.left,
                    format!("undefined variable for compound assignment: {}", name),
                ))
            }
        }
        // Compound assignment on an indexed field: a[i] += expr
        Expr::Index(_) => {
            let (left_reg, ty) = emit_expr(&bin.left, ctx)?;
            let (right_reg, _) = emit_expr(&bin.right, ctx)?;
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::BinOp {
                dst,
                a: left_reg,
                b: right_reg,
                op,
                ty,
            });
            emit_store_or_reassign(&bin.left, dst, ty, ctx)?;
            Ok(())
        }
        _ => Err(syn::Error::new_spanned(
            &bin.left,
            "compound assignment target must be a local variable or field[index]",
        )),
    }
}
