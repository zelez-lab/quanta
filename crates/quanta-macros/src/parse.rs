//! Parse a validated Rust function into KernelDef.
//!
//! Phase 2: full AST → KernelOp walking via recursive emit_expr/emit_stmt.

use quanta_ir::{
    AtomicOp, BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, MathFn, Reg, ScalarType,
    UnaryOp,
};
use std::collections::HashMap;
use syn::{BinOp as SynBinOp, Expr, FnArg, ItemFn, Pat, Stmt, Type, UnOp as SynUnOp};

/// Emission context — tracks registers, variables, and parameters.
struct EmitCtx {
    ops: Vec<KernelOp>,
    next_reg: u32,
    /// Variable name → (register, type)
    vars: HashMap<String, (Reg, ScalarType)>,
    /// Parameter name → (slot, kind, type)
    params: HashMap<String, ParamInfo>,
    /// Shared memory counter
    next_shared: u32,
}

#[derive(Clone)]
struct ParamInfo {
    slot: u32,
    is_const: bool,
    scalar_type: ScalarType,
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
        }
    }

    fn alloc_reg(&mut self) -> Reg {
        let r = Reg(self.next_reg);
        self.next_reg += 1;
        r
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

    for stmt in &func.block.stmts {
        emit_stmt(stmt, &mut ctx)?;
    }

    Ok(KernelDef {
        name,
        params,
        body: ctx.ops,
        body_source: None,
        next_reg: ctx.next_reg,
        opt_level: 3, // overridden by proc macro attribute
    })
}

// ============================================================================
// Statement emission
// ============================================================================

fn emit_stmt(stmt: &Stmt, ctx: &mut EmitCtx) -> Result<(), syn::Error> {
    match stmt {
        Stmt::Local(local) => {
            let var_name = match &local.pat {
                Pat::Ident(ident) => ident.ident.to_string(),
                _ => {
                    return Err(syn::Error::new_spanned(
                        &local.pat,
                        "unsupported pattern in let binding",
                    ));
                }
            };

            if let Some(init) = &local.init {
                let (reg, ty) = emit_expr(&init.expr, ctx)?;
                ctx.vars.insert(var_name, (reg, ty));
            }
            Ok(())
        }
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

/// Emit an expression used as a statement (e.g., assignment, function call).
fn emit_expr_stmt(expr: &Expr, ctx: &mut EmitCtx) -> Result<(), syn::Error> {
    match expr {
        // result[i] = value
        Expr::Assign(assign) => {
            let (src_reg, _) = emit_expr(&assign.right, ctx)?;
            emit_store(&assign.left, src_reg, ctx)?;
            Ok(())
        }
        // Compound assignment: result[i] += value
        Expr::Binary(bin) if is_assign_op(&bin.op) => {
            // Desugar a += b to a = a + b
            let (left_reg, ty) = emit_expr(&bin.left, ctx)?;
            let (right_reg, _) = emit_expr(&bin.right, ctx)?;
            let op = assign_op_to_binop(&bin.op)?;
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::BinOp {
                dst,
                a: left_reg,
                b: right_reg,
                op,
                ty,
            });
            emit_store(&bin.left, dst, ctx)?;
            Ok(())
        }
        // if/else as statement
        Expr::If(if_expr) => {
            emit_if(if_expr, ctx)?;
            Ok(())
        }
        // for loop
        Expr::ForLoop(for_loop) => {
            emit_for_loop(for_loop, ctx)?;
            Ok(())
        }
        // while loop
        Expr::While(while_loop) => {
            emit_while_loop(while_loop, ctx)?;
            Ok(())
        }
        // Expression with side effects (function calls like barrier())
        Expr::Call(call) => {
            emit_call(call, ctx)?;
            Ok(())
        }
        // Block expression
        Expr::Block(block) => {
            for stmt in &block.block.stmts {
                emit_stmt(stmt, ctx)?;
            }
            Ok(())
        }
        _ => {
            // Try as a general expression (discard result)
            emit_expr(expr, ctx)?;
            Ok(())
        }
    }
}

// ============================================================================
// Expression emission — returns (register, type)
// ============================================================================

fn emit_expr(expr: &Expr, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
    match expr {
        // Literal: 42, 0.99, true
        Expr::Lit(lit) => emit_literal(lit, ctx),

        // Variable reference: i, x, threshold
        Expr::Path(path) => emit_path(path, ctx),

        // Binary: a + b, x > threshold
        Expr::Binary(bin) => emit_binary(bin, ctx),

        // Unary: -x, !flag
        Expr::Unary(unary) => emit_unary(unary, ctx),

        // Index: a[i]
        Expr::Index(index) => emit_index(index, ctx),

        // Function call: quark_id(), sin(x), atomic_add(...)
        Expr::Call(call) => emit_call(call, ctx),

        // Method call: x.sin(), x.sqrt()
        Expr::MethodCall(mc) => emit_method_call(mc, ctx),

        // If expression: if cond { a } else { b }
        Expr::If(if_expr) => emit_if(if_expr, ctx),

        // Parenthesized: (a + b)
        Expr::Paren(paren) => emit_expr(&paren.expr, ctx),

        // Cast: x as f32
        Expr::Cast(cast) => emit_cast(cast, ctx),

        // Block: { stmts; final_expr }
        Expr::Block(block) => {
            let mut last = None;
            for stmt in &block.block.stmts {
                match stmt {
                    Stmt::Expr(e, None) => last = Some(emit_expr(e, ctx)?),
                    _ => {
                        emit_stmt(stmt, ctx)?;
                    }
                }
            }
            last.ok_or_else(|| syn::Error::new_spanned(expr, "empty block expression"))
        }

        _ => Err(syn::Error::new_spanned(
            expr,
            "unsupported expression in GPU kernel",
        )),
    }
}

fn emit_literal(lit: &syn::ExprLit, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
    let dst = ctx.alloc_reg();
    let (value, ty) = match &lit.lit {
        syn::Lit::Float(f) => {
            let v: f64 = f
                .base10_parse()
                .map_err(|e| syn::Error::new_spanned(f, e))?;
            // Check suffix
            let s = f.to_string();
            if s.ends_with("f32") || !s.ends_with("f64") {
                (ConstValue::F32(v as f32), ScalarType::F32)
            } else {
                (ConstValue::F64(v), ScalarType::F64)
            }
        }
        syn::Lit::Int(i) => {
            let s = i.to_string();
            if s.ends_with("u32") {
                let v: u32 = i
                    .base10_parse()
                    .map_err(|e| syn::Error::new_spanned(i, e))?;
                (ConstValue::U32(v), ScalarType::U32)
            } else if s.ends_with("u64") {
                let v: u64 = i
                    .base10_parse()
                    .map_err(|e| syn::Error::new_spanned(i, e))?;
                (ConstValue::U64(v), ScalarType::U64)
            } else if s.ends_with("i64") {
                let v: i64 = i
                    .base10_parse()
                    .map_err(|e| syn::Error::new_spanned(i, e))?;
                (ConstValue::I64(v), ScalarType::I64)
            } else {
                // Default integer → i32
                let v: i32 = i
                    .base10_parse()
                    .map_err(|e| syn::Error::new_spanned(i, e))?;
                (ConstValue::I32(v), ScalarType::I32)
            }
        }
        syn::Lit::Bool(b) => (ConstValue::Bool(b.value), ScalarType::Bool),
        _ => {
            return Err(syn::Error::new_spanned(
                &lit.lit,
                "unsupported literal type",
            ));
        }
    };
    ctx.ops.push(KernelOp::Const { dst, value });
    Ok((dst, ty))
}

fn emit_path(path: &syn::ExprPath, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
    let name = path
        .path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new_spanned(path, "empty path"))?
        .ident
        .to_string();

    // Check local variables first
    if let Some(&(reg, ty)) = ctx.vars.get(&name) {
        return Ok((reg, ty));
    }

    // Check push constant parameters
    if let Some(info) = ctx.params.get(&name).cloned()
        && info.is_const
    {
        let dst = ctx.alloc_reg();
        ctx.ops.push(KernelOp::Load {
            dst,
            field: info.slot,
            index: Reg(u32::MAX),
            ty: info.scalar_type,
        });
        return Ok((dst, info.scalar_type));
    }

    Err(syn::Error::new_spanned(
        path,
        format!("undefined variable: {}", name),
    ))
}

fn emit_binary(bin: &syn::ExprBinary, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
    let (a, ty_a) = emit_expr(&bin.left, ctx)?;
    let (b, _ty_b) = emit_expr(&bin.right, ctx)?;
    let dst = ctx.alloc_reg();

    // Check if it's a comparison
    if let Some(cmp) = syn_binop_to_cmp(&bin.op) {
        ctx.ops.push(KernelOp::Cmp {
            dst,
            a,
            b,
            op: cmp,
            ty: ty_a,
        });
        return Ok((dst, ScalarType::Bool));
    }

    // Arithmetic/bitwise
    let op = syn_binop_to_ir(&bin.op)?;
    ctx.ops.push(KernelOp::BinOp {
        dst,
        a,
        b,
        op,
        ty: ty_a,
    });
    Ok((dst, ty_a))
}

fn emit_unary(unary: &syn::ExprUnary, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
    let (a, ty) = emit_expr(&unary.expr, ctx)?;
    let dst = ctx.alloc_reg();
    let op = match unary.op {
        SynUnOp::Neg(_) => UnaryOp::Neg,
        SynUnOp::Not(_) => {
            if ty == ScalarType::Bool {
                UnaryOp::LogicalNot
            } else {
                UnaryOp::BitNot
            }
        }
        _ => return Err(syn::Error::new_spanned(unary, "unsupported unary operator")),
    };
    ctx.ops.push(KernelOp::UnaryOp { dst, a, op, ty });
    Ok((dst, ty))
}

fn emit_index(index: &syn::ExprIndex, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
    // arr[idx] — arr must be a parameter (field)
    let arr_name = expr_to_name(&index.expr).ok_or_else(|| {
        syn::Error::new_spanned(&index.expr, "indexing target must be a parameter name")
    })?;

    let info = ctx
        .params
        .get(&arr_name)
        .ok_or_else(|| {
            syn::Error::new_spanned(&index.expr, format!("unknown field: {}", arr_name))
        })?
        .clone();

    let (idx_reg, _) = emit_expr(&index.index, ctx)?;
    let dst = ctx.alloc_reg();
    ctx.ops.push(KernelOp::Load {
        dst,
        field: info.slot,
        index: idx_reg,
        ty: info.scalar_type,
    });
    Ok((dst, info.scalar_type))
}

fn emit_call(call: &syn::ExprCall, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
    let func_name = expr_to_name(&call.func).ok_or_else(|| {
        syn::Error::new_spanned(&call.func, "function call must be a simple name")
    })?;

    match func_name.as_str() {
        // Thread indexing
        "quark_id" => {
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::QuarkId { dst });
            Ok((dst, ScalarType::U32))
        }
        "quark_count" => {
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::QuarkCount { dst });
            Ok((dst, ScalarType::U32))
        }
        "local_id" => {
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::LocalId { dst });
            Ok((dst, ScalarType::U32))
        }
        "group_id" => {
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::GroupId { dst });
            Ok((dst, ScalarType::U32))
        }
        "group_size" => {
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::GroupSize { dst });
            Ok((dst, ScalarType::U32))
        }

        // Synchronization
        "barrier" => {
            ctx.ops.push(KernelOp::Barrier);
            Ok((Reg(u32::MAX), ScalarType::Bool))
        }

        // Atomics: atomic_add(&mut arr[i], val)
        "atomic_add" | "atomic_sub" | "atomic_min" | "atomic_max" | "atomic_and" | "atomic_or"
        | "atomic_xor" | "atomic_exchange" => emit_atomic_call(&func_name, call, ctx),

        // Math: sin(x), cos(x), sqrt(x), etc.
        _ => {
            if let Some(math_fn) = name_to_math_fn(&func_name) {
                let mut args = Vec::new();
                let mut ty = ScalarType::F32;
                for arg in &call.args {
                    let (r, t) = emit_expr(arg, ctx)?;
                    args.push(r);
                    ty = t;
                }
                let dst = ctx.alloc_reg();
                ctx.ops.push(KernelOp::MathCall {
                    dst,
                    func: math_fn,
                    args,
                    ty,
                });
                Ok((dst, ty))
            } else {
                Err(syn::Error::new_spanned(
                    &call.func,
                    format!("unknown GPU function: {}", func_name),
                ))
            }
        }
    }
}

fn emit_atomic_call(
    name: &str,
    call: &syn::ExprCall,
    ctx: &mut EmitCtx,
) -> Result<(Reg, ScalarType), syn::Error> {
    // atomic_add(&mut arr[i], val)
    if call.args.len() != 2 {
        return Err(syn::Error::new_spanned(
            call,
            format!("{} requires 2 arguments", name),
        ));
    }

    // First arg: &mut arr[idx]
    let target = &call.args[0];
    let (field_slot, idx_reg, ty) = parse_atomic_target(target, ctx)?;

    // Second arg: value
    let (val_reg, _) = emit_expr(&call.args[1], ctx)?;

    let op = match name {
        "atomic_add" => AtomicOp::Add,
        "atomic_sub" => AtomicOp::Sub,
        "atomic_min" => AtomicOp::Min,
        "atomic_max" => AtomicOp::Max,
        "atomic_and" => AtomicOp::And,
        "atomic_or" => AtomicOp::Or,
        "atomic_xor" => AtomicOp::Xor,
        "atomic_exchange" => AtomicOp::Exchange,
        _ => return Err(syn::Error::new_spanned(call, "unknown atomic op")),
    };

    let dst = ctx.alloc_reg();
    ctx.ops.push(KernelOp::AtomicOp {
        dst,
        field: field_slot,
        index: idx_reg,
        val: val_reg,
        op,
        ty,
    });
    Ok((dst, ty))
}

fn parse_atomic_target(
    expr: &Expr,
    ctx: &mut EmitCtx,
) -> Result<(u32, Reg, ScalarType), syn::Error> {
    // Expect &mut arr[idx]
    match expr {
        Expr::Reference(ref_expr) => match ref_expr.expr.as_ref() {
            Expr::Index(index) => {
                let arr_name = expr_to_name(&index.expr).ok_or_else(|| {
                    syn::Error::new_spanned(&index.expr, "atomic target must be a field")
                })?;
                let info = ctx
                    .params
                    .get(&arr_name)
                    .ok_or_else(|| {
                        syn::Error::new_spanned(&index.expr, format!("unknown field: {}", arr_name))
                    })?
                    .clone();
                let (idx_reg, _) = emit_expr(&index.index, ctx)?;
                Ok((info.slot, idx_reg, info.scalar_type))
            }
            _ => Err(syn::Error::new_spanned(
                expr,
                "atomic target must be &mut field[index]",
            )),
        },
        _ => Err(syn::Error::new_spanned(
            expr,
            "atomic target must be &mut field[index]",
        )),
    }
}

fn emit_method_call(
    mc: &syn::ExprMethodCall,
    ctx: &mut EmitCtx,
) -> Result<(Reg, ScalarType), syn::Error> {
    let method = mc.method.to_string();

    // x.sin(), x.cos(), x.sqrt(), x.abs()
    if let Some(math_fn) = name_to_math_fn(&method) {
        let (receiver, ty) = emit_expr(&mc.receiver, ctx)?;
        let mut args = vec![receiver];
        for arg in &mc.args {
            let (r, _) = emit_expr(arg, ctx)?;
            args.push(r);
        }
        let dst = ctx.alloc_reg();
        ctx.ops.push(KernelOp::MathCall {
            dst,
            func: math_fn,
            args,
            ty,
        });
        Ok((dst, ty))
    } else {
        Err(syn::Error::new_spanned(
            &mc.method,
            format!("unknown GPU method: {}", method),
        ))
    }
}

fn emit_cast(cast: &syn::ExprCast, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
    let (src, from) = emit_expr(&cast.expr, ctx)?;
    let to = match cast.ty.as_ref() {
        Type::Path(path) => scalar_type_from_path(path)?,
        _ => return Err(syn::Error::new_spanned(&cast.ty, "unsupported cast target")),
    };
    let dst = ctx.alloc_reg();
    ctx.ops.push(KernelOp::Cast { dst, src, from, to });
    Ok((dst, to))
}

fn emit_if(if_expr: &syn::ExprIf, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
    let (cond_reg, _) = emit_expr(&if_expr.cond, ctx)?;

    // Then branch
    let mut then_ctx = EmitCtx {
        ops: Vec::new(),
        next_reg: ctx.next_reg,
        vars: ctx.vars.clone(),
        params: ctx.params.clone(),
        next_shared: ctx.next_shared,
    };
    for stmt in &if_expr.then_branch.stmts {
        emit_stmt(stmt, &mut then_ctx)?;
    }
    let then_ops = then_ctx.ops;
    ctx.next_reg = then_ctx.next_reg;

    // Else branch
    let mut else_ops = Vec::new();
    if let Some((_, else_expr)) = &if_expr.else_branch {
        let mut else_ctx = EmitCtx {
            ops: Vec::new(),
            next_reg: ctx.next_reg,
            vars: ctx.vars.clone(),
            params: ctx.params.clone(),
            next_shared: ctx.next_shared,
        };
        match else_expr.as_ref() {
            Expr::Block(block) => {
                for stmt in &block.block.stmts {
                    emit_stmt(stmt, &mut else_ctx)?;
                }
            }
            Expr::If(nested_if) => {
                let _ = emit_if(nested_if, &mut else_ctx)?;
            }
            _ => {
                emit_expr_stmt(else_expr, &mut else_ctx)?;
            }
        }
        else_ops = else_ctx.ops;
        ctx.next_reg = else_ctx.next_reg;
    }

    ctx.ops.push(KernelOp::Branch {
        cond: cond_reg,
        then_ops,
        else_ops,
    });
    // If used as expression, we'd need phi — for now return a dummy
    Ok((Reg(u32::MAX), ScalarType::Bool))
}

fn emit_for_loop(for_loop: &syn::ExprForLoop, ctx: &mut EmitCtx) -> Result<(), syn::Error> {
    // for i in 0..N { body }
    let iter_name = match &*for_loop.pat {
        Pat::Ident(ident) => ident.ident.to_string(),
        _ => {
            return Err(syn::Error::new_spanned(
                &for_loop.pat,
                "for loop variable must be a simple name",
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
    ctx.vars.insert(iter_name, (iter_reg, ScalarType::U32));

    // Body
    let mut body_ctx = EmitCtx {
        ops: Vec::new(),
        next_reg: ctx.next_reg,
        vars: ctx.vars.clone(),
        params: ctx.params.clone(),
        next_shared: ctx.next_shared,
    };
    for stmt in &for_loop.body.stmts {
        emit_stmt(stmt, &mut body_ctx)?;
    }
    let body_ops = body_ctx.ops;
    ctx.next_reg = body_ctx.next_reg;

    ctx.ops.push(KernelOp::Loop {
        count: count_reg,
        iter_reg,
        body: body_ops,
    });
    Ok(())
}

fn emit_while_loop(while_loop: &syn::ExprWhile, _ctx: &mut EmitCtx) -> Result<(), syn::Error> {
    // while cond { body } — emit as Loop with a large count + break on !cond
    // Simplified: emit as a bounded loop (GPU kernels must be bounded)
    Err(syn::Error::new_spanned(
        while_loop,
        "while loops not yet supported — use for loops with bounded ranges",
    ))
}

/// Emit a store to a field: field[index] = value
fn emit_store(target: &Expr, src_reg: Reg, ctx: &mut EmitCtx) -> Result<(), syn::Error> {
    match target {
        Expr::Index(index) => {
            let arr_name = expr_to_name(&index.expr).ok_or_else(|| {
                syn::Error::new_spanned(&index.expr, "store target must be a field name")
            })?;
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
        _ => Err(syn::Error::new_spanned(
            target,
            "store target must be field[index]",
        )),
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn expr_to_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(path) => path.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

fn syn_binop_to_ir(op: &SynBinOp) -> Result<BinOp, syn::Error> {
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

fn syn_binop_to_cmp(op: &SynBinOp) -> Option<CmpOp> {
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

fn assign_op_to_binop(op: &SynBinOp) -> Result<BinOp, syn::Error> {
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

fn name_to_math_fn(name: &str) -> Option<MathFn> {
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
// Parameter parsing (unchanged from Phase 1)
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

fn scalar_type_from_path(path: &syn::TypePath) -> Result<ScalarType, syn::Error> {
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
