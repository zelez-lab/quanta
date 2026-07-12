//! Scalar-type inference for `#[quanta::kernel]` params.
//!
//! After the WASM-route cutover (slice 5d) the legacy parser's body
//! translator (`parse/{stmt,expr}.rs`) became dead-output code, but
//! `swap_body_via_wasm_route` still needed the scalar types on each
//! `KernelParam` — those drive the wasm-twin `*const T`/`*mut T`/
//! `T` param emission and seed the WASM lowerer's SideTable.
//!
//! This module is the focused replacement: it walks the kernel's syn
//! AST and produces a typed `Vec<KernelParam>` without emitting any
//! ops, allocating any registers, or dragging in `EmitCtx`. The
//! algorithm:
//!
//!   1. **Build the param list.** For struct-ref kernels, scan field
//!      accesses via `kernel_signature::scan_struct_field_accesses`
//!      and assign slots (buffers first, then scalars). For flat-
//!      param kernels, parse each fn arg's declared type.
//!   2. **Track local types.** Walk every `let NAME[: TY] = EXPR;`
//!      statement and record `NAME -> ScalarType` in a flow map.
//!      `let mut` and re-assignments via `ASSIGN`/compound-assign
//!      update the entry. Statements before any cast or annotation
//!      use literal-suffix inference (`0u32` → U32, `1.0f32` → F32).
//!   3. **Refine struct-ref buffer types.** For each statement of
//!      the form `d.field[idx] = EXPR`, look up `EXPR`'s inferred
//!      type and update `params[<field>].scalar_type` to match.
//!      Mirrors Path A (roadmap step 080) in `parse/stmt.rs:561`.
//!   4. **Refine struct-ref scalar (push-const) types.** Reuses the
//!      existing `parse::infer_const_scalar_type` walker — it covers
//!      the four hint shapes (type annotation, cast on opposite side
//!      of binop, literal on opposite side, direct cast).
//!
//! What this module does NOT cover (because nothing in the workspace
//! today exercises it): inferring buffer types from PURE READ
//! patterns (no write back). The legacy parser's body walker also
//! refined types via `retypecast_load_chain_to_int` and similar
//! op-emission side effects; those required tracking register types
//! through arithmetic. With the WASM route, that level of inference
//! isn't load-bearing — rustc's monomorphization picks the right
//! types from the user's source, and the WASM signature carries the
//! resolved widths into the lowerer.

use std::collections::HashMap;

use quanta_ir::{KernelDef, KernelParam, ScalarType};
use syn::{BinOp as SynBinOp, Expr, FnArg, ItemFn, Pat, Stmt, Type};

use crate::kernel_signature::{
    StructFieldAccess, StructRefParam, detect_struct_ref_param, scan_struct_field_accesses,
};
use crate::parse::{infer_const_scalar_type, parse_param_type, scalar_type_from_path};

/// Build a typed `KernelDef` with empty body — the body is filled in
/// downstream by `swap_body_via_wasm_route`. Returns the same shape
/// the legacy `parse::parse_kernel` produced for the params field;
/// the `body`, `next_reg`, `device_sources`, and `device_functions`
/// fields are placeholder-empty.
pub(crate) fn infer_kernel(func: &ItemFn) -> Result<KernelDef, syn::Error> {
    let name = func.sig.ident.to_string();
    let mut params = Vec::new();
    let mut slot = 0u32;

    let struct_ref = detect_struct_ref_param(func);

    if let Some(ref sr) = struct_ref {
        let field_accesses = scan_struct_field_accesses(func, &sr.param_name);
        for access in &field_accesses {
            params.push(initial_struct_ref_param(func, sr, access, slot));
            slot += 1;
        }
    } else {
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

    // Const generics → push constants.
    for generic in &func.sig.generics.params {
        if let syn::GenericParam::Const(cp) = generic {
            params.push(KernelParam::Constant {
                name: cp.ident.to_string(),
                slot,
                scalar_type: ScalarType::U32,
            });
            slot += 1;
        }
    }

    if let Some(sr) = struct_ref.as_ref() {
        refine_struct_ref_buffer_types(func, sr, &mut params);
        refine_struct_ref_scalar_types(func, sr, &mut params);
    }

    Ok(KernelDef {
        name,
        params,
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    })
}

fn initial_struct_ref_param(
    func: &ItemFn,
    sr: &StructRefParam,
    access: &StructFieldAccess,
    slot: u32,
) -> KernelParam {
    if access.is_indexed {
        // Buffer field: read-vs-write determined by access pattern.
        // Scalar type is a placeholder (F32) refined by the post-pass
        // walking `d.field[idx] = expr` assignments.
        if access.is_written {
            KernelParam::FieldWrite {
                name: access.name.clone(),
                slot,
                scalar_type: ScalarType::F32,
            }
        } else {
            KernelParam::FieldRead {
                name: access.name.clone(),
                slot,
                scalar_type: ScalarType::F32,
            }
        }
    } else {
        // Push-const scalar field. Use the existing inference walker
        // which examines type annotations / casts / literal suffixes
        // around `d.field` reads. Falls back to U32 (the historical
        // default) when no hint is found.
        let inferred =
            infer_const_scalar_type(func, &sr.param_name, &access.name).unwrap_or(ScalarType::U32);
        KernelParam::Constant {
            name: access.name.clone(),
            slot,
            scalar_type: inferred,
        }
    }
}

/// Walk the function body once to track local types, then walk a
/// second time to find `d.field[i] = expr` assignments and update
/// the matching buffer param's scalar type to match `expr`'s
/// inferred type.
fn refine_struct_ref_buffer_types(func: &ItemFn, sr: &StructRefParam, params: &mut [KernelParam]) {
    let mut locals: HashMap<String, ScalarType> = HashMap::new();
    for stmt in &func.block.stmts {
        track_locals_in_stmt(stmt, &mut locals);
    }

    for stmt in &func.block.stmts {
        refine_in_stmt(stmt, sr, &locals, params);
    }
}

/// Refine scalar (push-const) field types using the locals map. The
/// `parse::infer_const_scalar_type` heuristic already covers (a) type
/// annotations, (b) cast on the other side of a binop, (c) literal
/// suffixes, (d) direct casts. Those are encoded in the *initial*
/// param construction. This pass adds (e) "binop with a known local
/// on the other side" — e.g., `idx % d.width` where `idx: u32` infers
/// `d.width: u32`.
///
/// Important: the cast-applied-to-the-field heuristic in
/// `infer_const_scalar_type` (case 4) gives the WRONG answer when the
/// kernel writes `d.width as f32` — it sets the field type to f32
/// (the cast destination) instead of leaving room for the actual
/// source type. This pass overrides such mis-inferences when a
/// stronger signal (local-type binop) is available.
fn refine_struct_ref_scalar_types(func: &ItemFn, sr: &StructRefParam, params: &mut [KernelParam]) {
    let mut locals: HashMap<String, ScalarType> = HashMap::new();
    for stmt in &func.block.stmts {
        track_locals_in_stmt(stmt, &mut locals);
    }
    for stmt in &func.block.stmts {
        refine_scalar_in_stmt(stmt, sr, &locals, params);
    }
}

fn refine_scalar_in_stmt(
    stmt: &Stmt,
    sr: &StructRefParam,
    locals: &HashMap<String, ScalarType>,
    params: &mut [KernelParam],
) {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                refine_scalar_in_expr(&init.expr, sr, locals, params);
            }
        }
        Stmt::Expr(e, _) => refine_scalar_in_expr(e, sr, locals, params),
        _ => {}
    }
}

fn refine_scalar_in_expr(
    expr: &Expr,
    sr: &StructRefParam,
    locals: &HashMap<String, ScalarType>,
    params: &mut [KernelParam],
) {
    if let Expr::Binary(b) = expr {
        // `d.field op other` or `other op d.field`: if `other` has a
        // known scalar type via locals or literal suffix, use it.
        if let Some(field) = bare_struct_field(&b.left, sr)
            && let Some(ty) = infer_expr_type(&b.right, locals)
        {
            update_scalar_field(params, &field, ty);
        }
        if let Some(field) = bare_struct_field(&b.right, sr)
            && let Some(ty) = infer_expr_type(&b.left, locals)
        {
            update_scalar_field(params, &field, ty);
        }
    }
    // Recurse into sub-expressions.
    match expr {
        Expr::Binary(b) => {
            refine_scalar_in_expr(&b.left, sr, locals, params);
            refine_scalar_in_expr(&b.right, sr, locals, params);
        }
        Expr::Unary(u) => refine_scalar_in_expr(&u.expr, sr, locals, params),
        Expr::Paren(p) => refine_scalar_in_expr(&p.expr, sr, locals, params),
        Expr::Cast(c) => refine_scalar_in_expr(&c.expr, sr, locals, params),
        Expr::Reference(r) => refine_scalar_in_expr(&r.expr, sr, locals, params),
        Expr::Index(i) => {
            refine_scalar_in_expr(&i.expr, sr, locals, params);
            refine_scalar_in_expr(&i.index, sr, locals, params);
        }
        Expr::Call(c) => {
            for a in &c.args {
                refine_scalar_in_expr(a, sr, locals, params);
            }
        }
        Expr::MethodCall(m) => {
            refine_scalar_in_expr(&m.receiver, sr, locals, params);
            for a in &m.args {
                refine_scalar_in_expr(a, sr, locals, params);
            }
        }
        Expr::Assign(a) => {
            refine_scalar_in_expr(&a.left, sr, locals, params);
            refine_scalar_in_expr(&a.right, sr, locals, params);
        }
        Expr::Block(b) => {
            for s in &b.block.stmts {
                refine_scalar_in_stmt(s, sr, locals, params);
            }
        }
        Expr::If(if_expr) => {
            refine_scalar_in_expr(&if_expr.cond, sr, locals, params);
            for s in &if_expr.then_branch.stmts {
                refine_scalar_in_stmt(s, sr, locals, params);
            }
            if let Some((_, else_expr)) = &if_expr.else_branch {
                refine_scalar_in_expr(else_expr, sr, locals, params);
            }
        }
        Expr::While(w) => {
            refine_scalar_in_expr(&w.cond, sr, locals, params);
            for s in &w.body.stmts {
                refine_scalar_in_stmt(s, sr, locals, params);
            }
        }
        Expr::ForLoop(f) => {
            refine_scalar_in_expr(&f.expr, sr, locals, params);
            for s in &f.body.stmts {
                refine_scalar_in_stmt(s, sr, locals, params);
            }
        }
        _ => {}
    }
}

/// Return the field name if `expr` is `<sr.param>.NAME` (no index).
fn bare_struct_field(expr: &Expr, sr: &StructRefParam) -> Option<String> {
    let Expr::Field(f) = expr else {
        return None;
    };
    let Expr::Path(p) = f.base.as_ref() else {
        return None;
    };
    let seg = p.path.segments.last()?;
    if seg.ident != sr.param_name.as_str() {
        return None;
    }
    let syn::Member::Named(name) = &f.member else {
        return None;
    };
    Some(name.to_string())
}

/// Update only the `Constant` (push-const) variant — buffer-typed
/// fields are handled by `refine_struct_ref_buffer_types`.
fn update_scalar_field(params: &mut [KernelParam], field: &str, ty: ScalarType) {
    for p in params.iter_mut() {
        if let KernelParam::Constant {
            name, scalar_type, ..
        } = p
            && name == field
        {
            *scalar_type = ty;
            return;
        }
    }
}

fn track_locals_in_stmt(stmt: &Stmt, locals: &mut HashMap<String, ScalarType>) {
    match stmt {
        Stmt::Local(local) => {
            // `let [mut] NAME[: TY] = EXPR;`. Annotation wins; else
            // infer from the init expression.
            let (name, ty_annot) = match &local.pat {
                Pat::Ident(id) => (id.ident.to_string(), None),
                Pat::Type(pt) => match pt.pat.as_ref() {
                    Pat::Ident(id) => (id.ident.to_string(), Some(pt.ty.as_ref())),
                    _ => return,
                },
                _ => return,
            };
            let ty = if let Some(t) = ty_annot {
                ty_to_scalar(t)
            } else if let Some(init) = &local.init {
                infer_expr_type(&init.expr, locals)
            } else {
                None
            };
            if let Some(ty) = ty {
                locals.insert(name, ty);
            }
        }
        Stmt::Expr(expr, _) => {
            track_locals_in_expr(expr, locals);
        }
        _ => {}
    }
}

fn track_locals_in_expr(expr: &Expr, locals: &mut HashMap<String, ScalarType>) {
    match expr {
        Expr::Block(b) => {
            for s in &b.block.stmts {
                track_locals_in_stmt(s, locals);
            }
        }
        Expr::If(if_expr) => {
            for s in &if_expr.then_branch.stmts {
                track_locals_in_stmt(s, locals);
            }
            if let Some((_, else_expr)) = &if_expr.else_branch {
                track_locals_in_expr(else_expr, locals);
            }
        }
        Expr::While(w) => {
            for s in &w.body.stmts {
                track_locals_in_stmt(s, locals);
            }
        }
        Expr::ForLoop(f) => {
            for s in &f.body.stmts {
                track_locals_in_stmt(s, locals);
            }
        }
        _ => {}
    }
}

fn refine_in_stmt(
    stmt: &Stmt,
    sr: &StructRefParam,
    locals: &HashMap<String, ScalarType>,
    params: &mut [KernelParam],
) {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                refine_in_expr(&init.expr, sr, locals, params);
            }
        }
        Stmt::Expr(expr, _) => {
            refine_in_expr(expr, sr, locals, params);
        }
        _ => {}
    }
}

fn refine_in_expr(
    expr: &Expr,
    sr: &StructRefParam,
    locals: &HashMap<String, ScalarType>,
    params: &mut [KernelParam],
) {
    match expr {
        // `d.field[idx] = expr` — refine via the RHS type.
        Expr::Assign(assign) => {
            if let Some(field) = struct_index_field(&assign.left, sr)
                && let Some(rhs_ty) = infer_expr_type(&assign.right, locals)
            {
                set_field_scalar(params, &field, rhs_ty);
            }
            refine_in_expr(&assign.left, sr, locals, params);
            refine_in_expr(&assign.right, sr, locals, params);
        }
        // Compound assign `d.field[idx] += expr` — same refinement.
        Expr::Binary(bin) if is_assign_op(&bin.op) => {
            if let Some(field) = struct_index_field(&bin.left, sr)
                && let Some(rhs_ty) = infer_expr_type(&bin.right, locals)
            {
                set_field_scalar(params, &field, rhs_ty);
            }
            refine_in_expr(&bin.left, sr, locals, params);
            refine_in_expr(&bin.right, sr, locals, params);
        }
        Expr::Block(b) => {
            for s in &b.block.stmts {
                refine_in_stmt(s, sr, locals, params);
            }
        }
        Expr::If(if_expr) => {
            refine_in_expr(&if_expr.cond, sr, locals, params);
            for s in &if_expr.then_branch.stmts {
                refine_in_stmt(s, sr, locals, params);
            }
            if let Some((_, else_expr)) = &if_expr.else_branch {
                refine_in_expr(else_expr, sr, locals, params);
            }
        }
        Expr::While(w) => {
            refine_in_expr(&w.cond, sr, locals, params);
            for s in &w.body.stmts {
                refine_in_stmt(s, sr, locals, params);
            }
        }
        Expr::ForLoop(f) => {
            refine_in_expr(&f.expr, sr, locals, params);
            for s in &f.body.stmts {
                refine_in_stmt(s, sr, locals, params);
            }
        }
        Expr::Binary(bin) => {
            refine_in_expr(&bin.left, sr, locals, params);
            refine_in_expr(&bin.right, sr, locals, params);
        }
        Expr::Unary(u) => refine_in_expr(&u.expr, sr, locals, params),
        Expr::Paren(p) => refine_in_expr(&p.expr, sr, locals, params),
        Expr::Cast(c) => refine_in_expr(&c.expr, sr, locals, params),
        Expr::Call(c) => {
            for a in &c.args {
                refine_in_expr(a, sr, locals, params);
            }
        }
        Expr::MethodCall(m) => {
            refine_in_expr(&m.receiver, sr, locals, params);
            for a in &m.args {
                refine_in_expr(a, sr, locals, params);
            }
        }
        Expr::Reference(r) => refine_in_expr(&r.expr, sr, locals, params),
        Expr::Index(i) => {
            refine_in_expr(&i.expr, sr, locals, params);
            refine_in_expr(&i.index, sr, locals, params);
        }
        _ => {}
    }
}

/// Return the field name if `expr` is `<sr.param>.NAME[idx]`.
fn struct_index_field(expr: &Expr, sr: &StructRefParam) -> Option<String> {
    let Expr::Index(idx) = expr else {
        return None;
    };
    let Expr::Field(f) = idx.expr.as_ref() else {
        return None;
    };
    let Expr::Path(p) = f.base.as_ref() else {
        return None;
    };
    let seg = p.path.segments.last()?;
    if seg.ident != sr.param_name.as_str() {
        return None;
    }
    let syn::Member::Named(name) = &f.member else {
        return None;
    };
    Some(name.to_string())
}

fn set_field_scalar(params: &mut [KernelParam], field: &str, ty: ScalarType) {
    for p in params.iter_mut() {
        let n = match p {
            KernelParam::FieldRead { name, .. }
            | KernelParam::FieldWrite { name, .. }
            | KernelParam::Constant { name, .. }
            | KernelParam::Texture2DRead { name, .. }
            | KernelParam::Texture2DWrite { name, .. }
            | KernelParam::Texture3DRead { name, .. } => name.clone(),
        };
        if n != field {
            continue;
        }
        match p {
            KernelParam::FieldRead { scalar_type, .. }
            | KernelParam::FieldWrite { scalar_type, .. }
            | KernelParam::Constant { scalar_type, .. }
            | KernelParam::Texture2DRead { scalar_type, .. }
            | KernelParam::Texture2DWrite { scalar_type, .. }
            | KernelParam::Texture3DRead { scalar_type, .. } => {
                *scalar_type = ty;
            }
        }
        return;
    }
}

/// Infer an expression's scalar type using the locals map. Returns
/// `None` when no hint is available (caller falls back to whatever
/// default the param type already carries).
fn infer_expr_type(expr: &Expr, locals: &HashMap<String, ScalarType>) -> Option<ScalarType> {
    match expr {
        Expr::Cast(c) => ty_to_scalar(&c.ty),
        Expr::Lit(l) => lit_to_scalar(&l.lit),
        Expr::Paren(p) => infer_expr_type(&p.expr, locals),
        Expr::Group(g) => infer_expr_type(&g.expr, locals),
        Expr::Unary(u) => infer_expr_type(&u.expr, locals),
        Expr::Binary(b) => {
            let l = infer_expr_type(&b.left, locals);
            let r = infer_expr_type(&b.right, locals);
            // Prefer a non-None side; if both, prefer the left (matches
            // Rust's type-inference quirk where the first operand
            // sets the binop's expected type when ambiguous).
            l.or(r)
        }
        Expr::Path(p) => {
            let seg = p.path.segments.last()?;
            locals.get(&seg.ident.to_string()).copied()
        }
        Expr::Call(c) => {
            // Common Quanta intrinsics returning known scalars. We
            // don't cover the full math family — those usually
            // appear *as RHS of a `let` whose target type is the
            // refining annotation*, so the local-track pass covers
            // them. Indices here are only the truly-unannotated
            // cases.
            let Expr::Path(path) = c.func.as_ref() else {
                return None;
            };
            let seg = path.path.segments.last()?;
            match seg.ident.to_string().as_str() {
                "quark_id" | "local_id" | "group_id" | "proton_id" | "nucleus_id"
                | "subgroup_id" | "subgroup_size" | "workgroup_size" | "proton_size" => {
                    Some(ScalarType::U32)
                }
                "sqrt" | "rsqrt" | "sin" | "cos" | "tan" | "exp" | "ln" | "fabs" | "abs"
                | "floor" | "ceil" | "round" | "fmin" | "fmax" | "powf" | "fma" | "clamp_f"
                | "sqrt_f32" | "rsqrt_f32" | "sin_f32" | "cos_f32" | "tan_f32" | "exp_f32"
                | "log_f32" | "abs_f32" | "floor_f32" | "ceil_f32" | "round_f32" | "min_f32"
                | "max_f32" | "fma_f32" | "pow_f32" | "clamp_f32" => Some(ScalarType::F32),
                _ => None,
            }
        }
        _ => None,
    }
}

fn ty_to_scalar(ty: &Type) -> Option<ScalarType> {
    match ty {
        Type::Path(p) => scalar_type_from_path(p).ok(),
        Type::Reference(r) => ty_to_scalar(&r.elem),
        Type::Paren(p) => ty_to_scalar(&p.elem),
        _ => None,
    }
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
            // Unsuffixed int literals are ambiguous; don't refine.
            _ => None,
        },
        syn::Lit::Bool(_) => Some(ScalarType::Bool),
        _ => None,
    }
}

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
