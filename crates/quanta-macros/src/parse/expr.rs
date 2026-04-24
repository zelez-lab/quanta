//! Expression emission — returns (register, type).

use quanta_ir::{AtomicOp, BinOp, ConstValue, KernelOp, Reg, ScalarType, UnaryOp};
use syn::{BinOp as SynBinOp, Expr, Stmt, Type, UnOp as SynUnOp};

use super::stmt::emit_stmt;
use super::{
    EmitCtx, expr_to_name, name_to_math_fn, scalar_type_from_path, syn_binop_to_cmp,
    syn_binop_to_ir,
};

pub(crate) fn emit_expr(expr: &Expr, ctx: &mut EmitCtx) -> Result<(Reg, ScalarType), syn::Error> {
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

        // Tuple: (expr1, expr2) — used in tuple destructuring RHS
        Expr::Tuple(_) => Err(syn::Error::new_spanned(
            expr,
            "tuple expression only supported in let binding RHS",
        )),

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
                // Default integer -> i32
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
    // Handle logical AND/OR with short-circuit semantics modeled as bitwise on bools
    match &bin.op {
        SynBinOp::And(_) => {
            let (a, _) = emit_expr(&bin.left, ctx)?;
            let (b, _) = emit_expr(&bin.right, ctx)?;
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::BinOp {
                dst,
                a,
                b,
                op: BinOp::BitAnd,
                ty: ScalarType::Bool,
            });
            return Ok((dst, ScalarType::Bool));
        }
        SynBinOp::Or(_) => {
            let (a, _) = emit_expr(&bin.left, ctx)?;
            let (b, _) = emit_expr(&bin.right, ctx)?;
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::BinOp {
                dst,
                a,
                b,
                op: BinOp::BitOr,
                ty: ScalarType::Bool,
            });
            return Ok((dst, ScalarType::Bool));
        }
        _ => {}
    }

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
    // arr[idx] — arr can be a parameter (field) or a shared variable
    let arr_name = expr_to_name(&index.expr).ok_or_else(|| {
        syn::Error::new_spanned(&index.expr, "indexing target must be a parameter name")
    })?;

    // Check shared variables first
    if let Some(&(shared_id, scalar_ty)) = ctx.shared_vars.get(&arr_name) {
        let (idx_reg, _) = emit_expr(&index.index, ctx)?;
        let dst = ctx.alloc_reg();
        ctx.ops.push(KernelOp::SharedLoad {
            dst,
            id: shared_id,
            index: idx_reg,
            ty: scalar_ty,
        });
        return Ok((dst, scalar_ty));
    }

    let info = ctx
        .params
        .get(&arr_name)
        .ok_or_else(|| {
            syn::Error::new_spanned(&index.expr, format!("unknown field: {}", arr_name))
        })?
        .clone();

    // Index can be any expression (including complex ones like i * 4 + 1)
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
        "proton_id" => {
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::ProtonId { dst });
            Ok((dst, ScalarType::U32))
        }
        "nucleus_id" => {
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::NucleusId { dst });
            Ok((dst, ScalarType::U32))
        }
        "proton_size" => {
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::ProtonSize { dst });
            Ok((dst, ScalarType::U32))
        }
        "subgroup_size" => {
            let dst = ctx.alloc_reg();
            ctx.ops.push(KernelOp::SubgroupSize { dst });
            Ok((dst, ScalarType::U32))
        }

        // GPU debug print
        "gpu_print" => {
            if call.args.len() != 1 {
                return Err(syn::Error::new_spanned(
                    call,
                    "gpu_print requires exactly 1 argument",
                ));
            }
            let (val, ty) = emit_expr(&call.args[0], ctx)?;
            ctx.ops.push(KernelOp::DebugPrint { src: val, ty });
            Ok((val, ty))
        }

        // Synchronization
        "barrier" => {
            ctx.ops.push(KernelOp::Barrier);
            Ok((Reg(u32::MAX), ScalarType::Bool))
        }

        // Texture access: texture_load_2d(tex, x, y), texture_sample_2d(tex, x, y)
        "texture_load_2d" | "texture_sample_2d" => {
            if call.args.len() != 3 {
                return Err(syn::Error::new_spanned(
                    call,
                    format!("{func_name} requires 3 arguments: (texture, x, y)"),
                ));
            }
            // First arg is the texture param — resolve to its slot
            let tex_name = expr_to_name(&call.args[0]).ok_or_else(|| {
                syn::Error::new_spanned(&call.args[0], "texture argument must be a name")
            })?;
            let tex_slot = ctx
                .param_slot(&tex_name)
                .ok_or_else(|| syn::Error::new_spanned(&call.args[0], "unknown texture param"))?;
            let (x_reg, _) = emit_expr(&call.args[1], ctx)?;
            let (y_reg, _) = emit_expr(&call.args[2], ctx)?;
            let dst = ctx.alloc_reg();
            if func_name == "texture_load_2d" {
                ctx.ops.push(KernelOp::TextureLoad2D {
                    dst,
                    texture: tex_slot,
                    x: x_reg,
                    y: y_reg,
                    ty: ScalarType::F32,
                });
            } else {
                ctx.ops.push(KernelOp::TextureSample2D {
                    dst,
                    texture: tex_slot,
                    x: x_reg,
                    y: y_reg,
                    ty: ScalarType::F32,
                });
            }
            Ok((dst, ScalarType::F32))
        }

        // Atomics: atomic_add(&mut arr[i], val)
        "atomic_add" | "atomic_sub" | "atomic_min" | "atomic_max" | "atomic_and" | "atomic_or"
        | "atomic_xor" | "atomic_exchange" => emit_atomic_call(&func_name, call, ctx),

        // Math: sin(x), cos(x), sqrt(x), etc.
        // Device functions: user-defined inner functions
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
            } else if let Some(device_fn) = ctx.device_fns.get(&func_name).cloned() {
                // Device function call: emit args, then a DeviceCall op.
                // For the KernelOp IR path, we emit individual arg registers
                // and a DeviceCall op that references the function by name.
                let mut arg_regs = Vec::new();
                for arg in &call.args {
                    let (r, _) = emit_expr(arg, ctx)?;
                    arg_regs.push(r);
                }
                let dst = ctx.alloc_reg();
                ctx.ops.push(KernelOp::DeviceCall {
                    dst,
                    func_name: func_name.clone(),
                    args: arg_regs,
                    ty: device_fn.return_type,
                });
                Ok((dst, device_fn.return_type))
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

    // x.saturating_add(y), x.saturating_sub(y)
    if method == "saturating_add" || method == "saturating_sub" {
        let (receiver, ty) = emit_expr(&mc.receiver, ctx)?;
        if mc.args.len() != 1 {
            return Err(syn::Error::new_spanned(
                &mc.method,
                format!("{} takes exactly 1 argument", method),
            ));
        }
        let (arg, _) = emit_expr(&mc.args[0], ctx)?;
        let dst = ctx.alloc_reg();
        let op = if method == "saturating_add" {
            BinOp::SatAdd
        } else {
            BinOp::SatSub
        };
        ctx.ops.push(KernelOp::BinOp {
            dst,
            a: receiver,
            b: arg,
            op,
            ty,
        });
        return Ok((dst, ty));
    }

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

pub(crate) fn emit_if(
    if_expr: &syn::ExprIf,
    ctx: &mut EmitCtx,
) -> Result<(Reg, ScalarType), syn::Error> {
    let (cond_reg, _) = emit_expr(&if_expr.cond, ctx)?;

    // Then branch
    let mut then_ctx = ctx.child();
    for stmt in &if_expr.then_branch.stmts {
        emit_stmt(stmt, &mut then_ctx)?;
    }
    let then_ops = ctx.merge_child(then_ctx);

    // Else branch
    let mut else_ops = Vec::new();
    if let Some((_, else_expr)) = &if_expr.else_branch {
        let mut else_ctx = ctx.child();
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
        else_ops = ctx.merge_child(else_ctx);
    }

    ctx.ops.push(KernelOp::Branch {
        cond: cond_reg,
        then_ops,
        else_ops,
    });
    // If used as expression, we'd need phi — for now return a dummy
    Ok((Reg(u32::MAX), ScalarType::Bool))
}

/// Emit an expression used as a statement (e.g., assignment, function call).
/// Shared between stmt.rs (for emit_expr_stmt) and here (for emit_if else branches).
pub(crate) fn emit_expr_stmt(expr: &Expr, ctx: &mut EmitCtx) -> Result<(), syn::Error> {
    match expr {
        // result[i] = value  OR  x = value (local variable reassignment)
        Expr::Assign(assign) => {
            let (src_reg, src_ty) = emit_expr(&assign.right, ctx)?;
            super::stmt::emit_store_or_reassign(&assign.left, src_reg, src_ty, ctx)?;
            Ok(())
        }
        // Compound assignment: result[i] += value  OR  x += value (local)
        Expr::Binary(bin) if super::is_assign_op(&bin.op) => {
            super::stmt::emit_compound_assign(bin, ctx)?;
            Ok(())
        }
        // if/else as statement
        Expr::If(if_expr) => {
            emit_if(if_expr, ctx)?;
            Ok(())
        }
        // for loop
        Expr::ForLoop(for_loop) => {
            super::stmt::emit_for_loop(for_loop, ctx)?;
            Ok(())
        }
        // while loop
        Expr::While(while_loop) => {
            super::stmt::emit_while_loop(while_loop, ctx)?;
            Ok(())
        }
        // break
        Expr::Break(_) => {
            ctx.ops.push(KernelOp::Break);
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
