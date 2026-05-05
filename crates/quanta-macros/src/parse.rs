//! Scalar-type inference helpers — what's left of the legacy syn-AST
//! kernel parser after the WASM-route cutover (slice 5e).
//!
//! Pre-cutover this file (plus `parse/stmt.rs` + `parse/expr.rs`,
//! together ~3000 LoC) translated kernel bodies to `KernelOp`s. The
//! WASM route now does that translation via `rustc → wasm32 →
//! lower`, so the body translator is gone. What stayed:
//!
//!   - `parse_param_type` — converts a fn arg's syn `Type` to a
//!     `KernelParam`. Used by `kernel_type_inference` for flat-param
//!     kernels and by anything else that needs to read the
//!     declared param shape.
//!   - `infer_const_scalar_type` — heuristic walker that infers a
//!     struct push-constant field's scalar type from usage hints
//!     (type annotations / casts / literal suffixes). Used by
//!     `kernel_type_inference::initial_struct_ref_param`.
//!   - `scalar_type_from_path` — the syn-Path → `ScalarType` lookup
//!     table.
//!
//! The struct-ref signature analysis (`detect_struct_ref_param`,
//! `scan_struct_field_accesses`, `StructRefParam`,
//! `StructFieldAccess`) lives in `crate::kernel_signature` (slice
//! 5a) — only re-exported here for any historical call site that
//! still uses `parse::detect_struct_ref_param` etc.

use quanta_ir::{KernelParam, ScalarType};
use syn::{Expr, ItemFn, Stmt, Type};

/// Infer the scalar type of a struct push-constant field by walking
/// the kernel body looking for usage-context hints. Returns `None`
/// when no hint can be derived; caller should fall back to the
/// historical default (`ScalarType::U32`).
///
/// The hints we recognise (in priority order):
/// 1. **Type annotation**: `let x: f32 = p.field;`
/// 2. **Cast on the other side of a binop**: `(i as f32) * p.field`
/// 3. **Literal on the other side of a binop**: `p.field * 2.0_f32`
/// 4. **`as` cast applied to the field directly**: `(p.field as f32)`
///
/// Heuristic only — doesn't promise to catch every case. When a hint
/// can't be derived, the default is preserved and the user can
/// either adjust their kernel or use the manual API.
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
            if let syn::Pat::Type(pat_ty) = &local.pat
                && let Some(init) = &local.init
                && expr_is_field(&init.expr, param_name, field_name)
                && let Some(ty) = ty_to_scalar(&pat_ty.ty)
            {
                *hint = Some(ty);
                return;
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
            let l_is = expr_is_field(&b.left, param_name, field_name);
            let r_is = expr_is_field(&b.right, param_name, field_name);
            if l_is && let Some(ty) = expr_to_scalar(&b.right) {
                *hint = Some(ty);
                return;
            }
            if r_is && let Some(ty) = expr_to_scalar(&b.left) {
                *hint = Some(ty);
                return;
            }
            walk_expr_for_const_hint(&b.left, param_name, field_name, hint);
            walk_expr_for_const_hint(&b.right, param_name, field_name, hint);
        }
        Expr::Cast(c) => {
            if expr_is_field(&c.expr, param_name, field_name)
                && let Some(ty) = ty_to_scalar(&c.ty)
            {
                *hint = Some(ty);
                return;
            }
            walk_expr_for_const_hint(&c.expr, param_name, field_name, hint);
        }
        Expr::Paren(p) => walk_expr_for_const_hint(&p.expr, param_name, field_name, hint),
        Expr::Group(g) => walk_expr_for_const_hint(&g.expr, param_name, field_name, hint),
        Expr::Unary(u) => walk_expr_for_const_hint(&u.expr, param_name, field_name, hint),
        Expr::Reference(r) => walk_expr_for_const_hint(&r.expr, param_name, field_name, hint),
        Expr::Index(i) => {
            walk_expr_for_const_hint(&i.expr, param_name, field_name, hint);
            walk_expr_for_const_hint(&i.index, param_name, field_name, hint);
        }
        Expr::Call(c) => {
            for a in &c.args {
                walk_expr_for_const_hint(a, param_name, field_name, hint);
                if hint.is_some() {
                    return;
                }
            }
        }
        Expr::MethodCall(m) => {
            walk_expr_for_const_hint(&m.receiver, param_name, field_name, hint);
            for a in &m.args {
                walk_expr_for_const_hint(a, param_name, field_name, hint);
                if hint.is_some() {
                    return;
                }
            }
        }
        Expr::Block(b) => walk_block_for_const_hint(&b.block, param_name, field_name, hint),
        Expr::If(if_expr) => {
            walk_expr_for_const_hint(&if_expr.cond, param_name, field_name, hint);
            walk_block_for_const_hint(&if_expr.then_branch, param_name, field_name, hint);
            if let Some((_, else_expr)) = &if_expr.else_branch {
                walk_expr_for_const_hint(else_expr, param_name, field_name, hint);
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
            walk_expr_for_const_hint(&a.left, param_name, field_name, hint);
            walk_expr_for_const_hint(&a.right, param_name, field_name, hint);
        }
        _ => {}
    }
}

/// Returns true if `expr` is exactly `param.field_name`, OR
/// `param.field_name[any_index]` (buffer indexed access).
fn expr_is_field(expr: &Expr, param_name: &str, field_name: &str) -> bool {
    if let Expr::Field(f) = expr
        && let Expr::Path(p) = f.base.as_ref()
        && let Some(seg) = p.path.segments.last()
        && seg.ident == param_name
        && let syn::Member::Named(ident) = &f.member
    {
        return ident == field_name;
    }
    if let Expr::Index(i) = expr
        && let Expr::Field(f) = i.expr.as_ref()
        && let Expr::Path(p) = f.base.as_ref()
        && let Some(seg) = p.path.segments.last()
        && seg.ident == param_name
        && let syn::Member::Named(ident) = &f.member
    {
        return ident == field_name;
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
            // unsuffixed int literal — ambiguous
            _ => None,
        },
        syn::Lit::Bool(_) => Some(ScalarType::Bool),
        _ => None,
    }
}

/// Convert a function-arg syn `Type` to a `KernelParam`. Slice and
/// texture references become buffer/texture slots; bare scalars
/// become push constants. Used by `kernel_type_inference` to seed
/// the param list for flat-param kernels.
pub(crate) fn parse_param_type(
    name: &str,
    ty: &Type,
    slot: u32,
) -> Result<KernelParam, syn::Error> {
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
                        if ident == "Texture3D" {
                            let scalar = extract_generic_scalar(seg)?;
                            return Ok(KernelParam::Texture3DRead {
                                name: name.to_string(),
                                slot,
                                scalar_type: scalar,
                            });
                        }
                    }
                    Err(syn::Error::new_spanned(
                        ty,
                        "reference parameter must be &[T], &Texture2D<T>, &Texture3D<T>, or scalar &T",
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
    Ok(ScalarType::F32)
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
