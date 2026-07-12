//! AST-driven MSL body emitter for vertex/fragment shaders.
//!
//! The shader `body_source` arrives as a token-stringified Rust block (from
//! the proc-macro's `to_token_stream().to_string()`), so `Vec4::new` may be
//! line-wrapped as `Vec4 :: new`, `if` sits in statement position, and `&T`
//! uniform params are plain values. The previous emitter string-replaced over
//! that text and broke on all three. This module re-parses the block with
//! `syn` and walks the real AST, so formatting is irrelevant and the emitted
//! MSL is structurally correct.
//!
//! # Parity with the SPIR-V emitter
//!
//! The construct surface mirrors `emit_spirv`'s recursive-descent shader
//! parser (`expr.rs` / `expr_atom.rs` / `tokenizer.rs`): the same Vec
//! constructors, GLSL.std.450 intrinsic set, `dot`, `sample(slot, uv)`,
//! swizzle/field access, arithmetic, comparisons, `if/else`, and `let`
//! bindings. Anything outside that surface is rejected here with a clear
//! `Err(String)` naming the construct — the caller (`shader_pipeline`) turns
//! that into a build error, so an unsupported shape fails loudly rather than
//! miscompiling.

use std::collections::HashMap;
use std::fmt::Write as _;

use syn::{BinOp, Expr, Lit, Stmt, UnOp};

/// The MSL value types a shader expression can have. Shaders are float-only, so
/// this is scalars + float vectors, plus `Bool` for comparison results.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MslType {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Bool,
    /// Type couldn't be inferred (an unknown identifier, an intrinsic whose
    /// result type we don't model). Only fatal where a concrete declaration is
    /// required (a `let x = if ...` whose value type must be named).
    Unknown,
}

impl MslType {
    fn name(self) -> &'static str {
        match self {
            MslType::Float => "float",
            MslType::Vec2 => "float2",
            MslType::Vec3 => "float3",
            MslType::Vec4 => "float4",
            MslType::Bool => "bool",
            MslType::Unknown => "auto",
        }
    }

    pub(crate) fn from_shader_type(ty: quanta_ir::ShaderType) -> MslType {
        match ty {
            quanta_ir::ShaderType::F32 => MslType::Float,
            quanta_ir::ShaderType::Vec2 => MslType::Vec2,
            quanta_ir::ShaderType::Vec3 => MslType::Vec3,
            quanta_ir::ShaderType::Vec4 => MslType::Vec4,
            // Matrices aren't values a shader body constructs; treat as unknown.
            quanta_ir::ShaderType::Mat4 | quanta_ir::ShaderType::Mat3 => MslType::Unknown,
        }
    }
}

/// Name → inferred MSL type for every param and in-scope `let` binding.
///
/// A shader body is a flat cascade of `let`s (plus nested branch blocks that
/// only add locals used within the branch), so a single growing map is enough —
/// branch-local names never escape their block, and no shader shadows a name.
type TypeEnv = HashMap<String, MslType>;

/// Intrinsics that map 1:1 to an MSL builtin of the same name.
///
/// Kept in lockstep with `emit_spirv::tokenizer::glsl_func_id` — every GLSL
/// ext-inst the SPIR-V side accepts has an MSL builtin of the same (or
/// aliased) spelling. `dot` is handled separately (SPIR-V `OpDot`, MSL `dot`).
fn intrinsic_msl_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "sin" => "sin",
        "cos" => "cos",
        "tan" => "tan",
        "asin" => "asin",
        "acos" => "acos",
        "atan" => "atan",
        "atan2" => "atan2",
        "sqrt" => "sqrt",
        "inverseSqrt" | "inverse_sqrt" => "rsqrt",
        "abs" => "abs",
        "floor" => "floor",
        "ceil" => "ceil",
        "round" => "round",
        "fract" => "fract",
        "min" => "min",
        "max" => "max",
        "clamp" => "clamp",
        "mix" => "mix",
        "step" => "step",
        "smoothstep" | "smooth_step" => "smoothstep",
        "pow" => "pow",
        "exp" => "exp",
        "log" => "log",
        "exp2" => "exp2",
        "log2" => "log2",
        "normalize" => "normalize",
        "length" => "length",
        "distance" => "distance",
        "cross" => "cross",
        "fma" => "fma",
        "dot" => "dot",
        _ => return None,
    })
}

/// Map a `VecN::new` constructor path to its MSL vector type name.
fn vec_ctor_msl(name: &str) -> Option<&'static str> {
    Some(match name {
        "Vec2" => "float2",
        "Vec3" => "float3",
        "Vec4" => "float4",
        _ => return None,
    })
}

/// Emit the MSL body of a shader block.
///
/// `assign_result_to` controls how the block's tail expression is bound:
/// - `Some(var)` — emit `<var> = <tail>;` (vertex: the position result feeds an
///   output struct member).
/// - `None` — emit `return <tail>;` (fragment: the tail is the output color).
///
/// Both forms handle a tail that is itself an `if/else` by lowering the branch
/// arms to assignments/returns.
pub(crate) fn emit_body(
    body_source: &str,
    assign_result_to: Option<&str>,
    params: &[(String, MslType)],
) -> Result<String, String> {
    let block: syn::Block = syn::parse_str(body_source)
        .map_err(|e| format!("failed to parse shader body as a Rust block: {e}"))?;
    let mut env: TypeEnv = params.iter().cloned().collect();
    let mut out = String::new();
    emit_stmts(&block.stmts, assign_result_to, 1, &mut env, &mut out)?;
    Ok(out)
}

/// Emit a statement sequence. The final expression statement (no trailing `;`)
/// is the block's value and is routed through `tail`.
fn emit_stmts(
    stmts: &[Stmt],
    tail: Option<&str>,
    indent: usize,
    env: &mut TypeEnv,
    out: &mut String,
) -> Result<(), String> {
    let pad = "    ".repeat(indent);
    for (i, stmt) in stmts.iter().enumerate() {
        let is_last = i == stmts.len() - 1;
        match stmt {
            Stmt::Local(local) => emit_local(local, indent, env, out)?,
            Stmt::Expr(expr, semi) => {
                if is_last && semi.is_none() {
                    emit_tail(expr, tail, indent, env, out)?;
                } else {
                    // A non-tail expression statement: only assignments carry
                    // meaning in a shader body (locals are the primary form).
                    let code = emit_expr(expr)?;
                    writeln!(out, "{pad}{code};").unwrap();
                }
            }
            other => {
                return Err(format!(
                    "unsupported statement in shader body: {}",
                    stmt_kind(other)
                ));
            }
        }
    }
    Ok(())
}

/// Emit the block's tail expression, routed to a return or an assignment.
///
/// A tail `if/else` is lowered structurally: each arm emits its own
/// return/assignment, so no MSL ternary or phi temp is needed.
fn emit_tail(
    expr: &Expr,
    tail: Option<&str>,
    indent: usize,
    env: &mut TypeEnv,
    out: &mut String,
) -> Result<(), String> {
    let pad = "    ".repeat(indent);
    if let Expr::If(if_expr) = expr {
        emit_if(if_expr, TailMode::Route(tail), indent, env, out)?;
        return Ok(());
    }
    let code = emit_expr(expr)?;
    match tail {
        Some(var) => writeln!(out, "{pad}{var} = {code};").unwrap(),
        None => writeln!(out, "{pad}return {code};").unwrap(),
    }
    Ok(())
}

/// Emit a `let` binding.
///
/// `let x = <expr>;` becomes `<ty> x = <expr>;` (the inferred concrete type, or
/// `auto` when a simple expression lets the compiler deduce it). A
/// `let x = if c {a} else {b};` declares `<ty> x;` with the branch value's
/// inferred type then lowers the `if` into assignments to `x` — MSL has no
/// if-expression, and an uninitialized `auto` is illegal, so the type must be
/// named.
fn emit_local(
    local: &syn::Local,
    indent: usize,
    env: &mut TypeEnv,
    out: &mut String,
) -> Result<(), String> {
    let pad = "    ".repeat(indent);
    let name = pat_ident(&local.pat)?;
    let init = local.init.as_ref().ok_or_else(|| {
        format!("`let {name}` without an initializer is not supported in shaders")
    })?;
    if init.diverge.is_some() {
        return Err(format!(
            "`let {name} = ... else` is not supported in shaders"
        ));
    }

    if let Expr::If(if_expr) = init.expr.as_ref() {
        // The declared type is the type of a branch's value. Infer it from the
        // then-branch tail (both branches share a type in well-formed Rust).
        let ty = infer_if_type(if_expr, env);
        if ty == MslType::Unknown {
            return Err(format!(
                "cannot infer the type of `let {name} = if ...` — the branch value type is needed to declare the MSL local"
            ));
        }
        env.insert(name.clone(), ty);
        writeln!(out, "{pad}{} {name};", ty.name()).unwrap();
        emit_if(if_expr, TailMode::Assign(&name), indent, env, out)?;
    } else {
        let ty = infer_type(&init.expr, env);
        let code = emit_expr(&init.expr)?;
        env.insert(name.clone(), ty);
        // `auto` is fine for an initialized binding; only if-lowered locals need
        // an explicit type. Keep `auto` to stay agnostic where inference is
        // unsure but the initializer makes it concrete for the MSL compiler.
        writeln!(out, "{pad}auto {name} = {code};").unwrap();
    }
    Ok(())
}

/// How an `if`'s branch tails are lowered.
#[derive(Clone, Copy)]
enum TailMode<'a> {
    /// Assign each branch's value to a pre-declared local.
    Assign(&'a str),
    /// Route each branch's value to the block tail (`return` or outer assign).
    Route(Option<&'a str>),
}

/// Emit an `if/else` as an MSL statement.
///
/// The condition is an ordinary boolean expression; each branch is a nested
/// block that may itself contain `let` bindings and a tail — the tail is
/// lowered per `mode`. Both branches are required (matches the SPIR-V side,
/// which needs an else for its phi); an `if` without `else` is only valid when
/// its own value is unused, which never happens for a shader tail.
fn emit_if(
    if_expr: &syn::ExprIf,
    mode: TailMode,
    indent: usize,
    env: &mut TypeEnv,
    out: &mut String,
) -> Result<(), String> {
    let pad = "    ".repeat(indent);
    let cond = emit_expr(&if_expr.cond)?;
    writeln!(out, "{pad}if ({cond}) {{").unwrap();
    emit_branch_block(&if_expr.then_branch.stmts, mode, indent + 1, env, out)?;

    match &if_expr.else_branch {
        Some((_, else_expr)) => {
            writeln!(out, "{pad}}} else {{").unwrap();
            match else_expr.as_ref() {
                // `else if` chains as a nested if inside the else block.
                Expr::If(nested) => emit_if(nested, mode, indent + 1, env, out)?,
                Expr::Block(block) => {
                    emit_branch_block(&block.block.stmts, mode, indent + 1, env, out)?
                }
                other => {
                    return Err(format!(
                        "unsupported else branch in shader `if`: {}",
                        expr_kind(other)
                    ));
                }
            }
            writeln!(out, "{pad}}}").unwrap();
        }
        None => {
            return Err("`if` without `else` is not supported in shader expressions".to_string());
        }
    }
    Ok(())
}

/// Emit the statements of an `if` branch block; the branch's tail expression is
/// lowered according to `mode` (assign to a local, or route to the block tail).
fn emit_branch_block(
    stmts: &[Stmt],
    mode: TailMode,
    indent: usize,
    env: &mut TypeEnv,
    out: &mut String,
) -> Result<(), String> {
    let tail = match mode {
        TailMode::Assign(v) => Some(v),
        TailMode::Route(t) => t,
    };
    emit_stmts(stmts, tail, indent, env, out)
}

/// Infer the MSL type an `if/else` evaluates to, from its then-branch tail.
///
/// The branch may declare its own `let`s before the tail (`{ let d = ..; d }`),
/// so inference threads a scratch env seeded from the outer scope through those
/// bindings before typing the tail — otherwise a tail that names a branch-local
/// would read as `Unknown`.
fn infer_if_type(if_expr: &syn::ExprIf, env: &TypeEnv) -> MslType {
    infer_branch_type(&if_expr.then_branch.stmts, env)
}

/// Infer the value type of a branch block by simulating its `let` bindings.
fn infer_branch_type(stmts: &[Stmt], outer: &TypeEnv) -> MslType {
    let mut scope = outer.clone();
    for (i, stmt) in stmts.iter().enumerate() {
        match stmt {
            Stmt::Local(local) => {
                if let (Ok(name), Some(init)) = (pat_ident(&local.pat), local.init.as_ref()) {
                    let ty = infer_type(&init.expr, &scope);
                    scope.insert(name, ty);
                }
            }
            Stmt::Expr(expr, None) if i == stmts.len() - 1 => {
                return infer_type(expr, &scope);
            }
            _ => {}
        }
    }
    MslType::Unknown
}

/// Infer the MSL type of an expression from structure. Unmodeled shapes return
/// `Unknown`; that's only fatal where a concrete type must be named (if-let).
fn infer_type(expr: &Expr, env: &TypeEnv) -> MslType {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Bool(_) => MslType::Bool,
            _ => MslType::Float,
        },
        Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|id| env.get(&id.to_string()).copied())
            .unwrap_or(MslType::Unknown),
        Expr::Paren(p) => infer_type(&p.expr, env),
        Expr::Group(g) => infer_type(&g.expr, env),
        Expr::Cast(c) => infer_type(&c.expr, env),
        Expr::Unary(u) => match u.op {
            UnOp::Not(_) => MslType::Bool,
            // Deref of a `&Vec2` uniform yields the element type; but we only
            // reach here for `*name`, whose element type is the param's type.
            _ => infer_type(&u.expr, env),
        },
        Expr::Binary(b) => infer_binary_type(b, env),
        Expr::Field(f) => infer_field_type(f),
        Expr::If(if_expr) => infer_if_type(if_expr, env),
        Expr::Call(c) => infer_call_type(c, env),
        _ => MslType::Unknown,
    }
}

fn infer_binary_type(b: &syn::ExprBinary, env: &TypeEnv) -> MslType {
    match b.op {
        BinOp::Lt(_)
        | BinOp::Gt(_)
        | BinOp::Le(_)
        | BinOp::Ge(_)
        | BinOp::Eq(_)
        | BinOp::Ne(_)
        | BinOp::And(_)
        | BinOp::Or(_) => MslType::Bool,
        // Arithmetic: the vector operand wins (scalar·vector = vector);
        // otherwise the left type.
        _ => {
            let l = infer_type(&b.left, env);
            let r = infer_type(&b.right, env);
            match (l, r) {
                (MslType::Float, other) | (MslType::Unknown, other) if other != MslType::Float => {
                    other
                }
                (l, _) => l,
            }
        }
    }
}

/// Swizzle/field result type: single component → float, N-char swizzle → floatN.
fn infer_field_type(f: &syn::ExprField) -> MslType {
    if let syn::Member::Named(id) = &f.member {
        return match id.to_string().len() {
            1 => MslType::Float,
            2 => MslType::Vec2,
            3 => MslType::Vec3,
            4 => MslType::Vec4,
            _ => MslType::Unknown,
        };
    }
    MslType::Unknown
}

fn infer_call_type(c: &syn::ExprCall, env: &TypeEnv) -> MslType {
    let Expr::Path(callee) = c.func.as_ref() else {
        return MslType::Unknown;
    };
    let path = &callee.path;
    // Vec constructors.
    if path.segments.len() == 2 && path.segments[1].ident == "new" {
        return match path.segments[0].ident.to_string().as_str() {
            "Vec2" => MslType::Vec2,
            "Vec3" => MslType::Vec3,
            "Vec4" => MslType::Vec4,
            _ => MslType::Unknown,
        };
    }
    let Some(name) = path.get_ident().map(|i| i.to_string()) else {
        return MslType::Unknown;
    };
    match name.as_str() {
        // `sample()` returns a full Vec4.
        "sample" => MslType::Vec4,
        // Reductions to a scalar.
        "length" | "distance" | "dot" => MslType::Float,
        // Component-wise intrinsics take the type of their first argument.
        "min" | "max" | "clamp" | "mix" | "abs" | "floor" | "ceil" | "round" | "fract"
        | "normalize" | "step" | "smoothstep" | "pow" | "sin" | "cos" | "tan" | "sqrt" | "exp"
        | "log" | "exp2" | "log2" | "fma" | "cross" => c
            .args
            .first()
            .map(|a| infer_type(a, env))
            .unwrap_or(MslType::Unknown),
        _ => MslType::Unknown,
    }
}

/// Emit an expression to an MSL source fragment (no trailing `;`).
fn emit_expr(expr: &Expr) -> Result<String, String> {
    match expr {
        Expr::Lit(lit) => emit_lit(&lit.lit),
        Expr::Path(path) => emit_path_ident(&path.path),
        Expr::Paren(p) => Ok(format!("({})", emit_expr(&p.expr)?)),
        Expr::Group(g) => emit_expr(&g.expr),
        Expr::Unary(u) => emit_unary(u),
        Expr::Binary(b) => emit_binary(b),
        // `x as f32` / `as u32` etc. — shaders are float-only, strip the cast.
        Expr::Cast(c) => emit_expr(&c.expr),
        Expr::Field(f) => emit_field(f),
        Expr::MethodCall(m) => emit_method_call(m),
        Expr::Call(c) => emit_call(c),
        Expr::If(_) => Err(
            "an `if` used as a value must be bound with `let x = if ...` or be the block tail"
                .to_string(),
        ),
        other => Err(format!(
            "unsupported expression in shader body: {}",
            expr_kind(other)
        )),
    }
}

fn emit_lit(lit: &Lit) -> Result<String, String> {
    match lit {
        Lit::Float(f) => {
            // Strip any Rust suffix (`1.0f32`) and guarantee a decimal point so
            // MSL types the literal as float, not int.
            let s = f.base10_digits();
            Ok(ensure_float_literal(s))
        }
        Lit::Int(i) => {
            let s = i.base10_digits();
            Ok(ensure_float_literal(s))
        }
        Lit::Bool(b) => Ok(if b.value { "true" } else { "false" }.to_string()),
        other => Err(format!("unsupported literal in shader body: {other:?}")),
    }
}

/// Ensure a numeric literal string reads as an MSL float (has a `.`).
fn ensure_float_literal(s: &str) -> String {
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s.to_string()
    } else {
        format!("{s}.0")
    }
}

/// A path used as a value: a bare identifier (`corner`, `viewport`) or a
/// boolean literal spelled as a path. Multi-segment paths are only valid as a
/// constructor callee (handled in `emit_call`), never as a bare value.
fn emit_path_ident(path: &syn::Path) -> Result<String, String> {
    if let Some(ident) = path.get_ident() {
        return Ok(ident.to_string());
    }
    Err(format!(
        "unsupported path expression in shader body: `{}`",
        path_to_string(path)
    ))
}

fn emit_unary(u: &syn::ExprUnary) -> Result<String, String> {
    let inner = emit_expr(&u.expr)?;
    match u.op {
        UnOp::Neg(_) => Ok(format!("-{inner}")),
        UnOp::Not(_) => Ok(format!("!{inner}")),
        // `*viewport` — a deref of a `&T` uniform. In MSL the uniform is bound
        // as `constant T&`, so the value IS the deref; drop the operator.
        UnOp::Deref(_) => Ok(inner),
        other => Err(format!(
            "unsupported unary operator in shader body: {other:?}"
        )),
    }
}

fn emit_binary(b: &syn::ExprBinary) -> Result<String, String> {
    let l = emit_expr(&b.left)?;
    let r = emit_expr(&b.right)?;
    let op = match b.op {
        BinOp::Add(_) => "+",
        BinOp::Sub(_) => "-",
        BinOp::Mul(_) => "*",
        BinOp::Div(_) => "/",
        BinOp::Rem(_) => "%",
        BinOp::Lt(_) => "<",
        BinOp::Gt(_) => ">",
        BinOp::Le(_) => "<=",
        BinOp::Ge(_) => ">=",
        BinOp::Eq(_) => "==",
        BinOp::Ne(_) => "!=",
        BinOp::And(_) => "&&",
        BinOp::Or(_) => "||",
        other => {
            return Err(format!(
                "unsupported binary operator in shader body: {other:?}"
            ));
        }
    };
    Ok(format!("{l} {op} {r}"))
}

/// Field / swizzle access. `.x/.y/.z/.w` and `.r/.g/.b/.a` map straight through
/// (MSL uses the same swizzle spellings); multi-char swizzles (`.xy`, `.zw`)
/// are supported too since the runtime shaders use `uv_rect.zw`.
fn emit_field(f: &syn::ExprField) -> Result<String, String> {
    let base = emit_expr(&f.base)?;
    let member = match &f.member {
        syn::Member::Named(id) => id.to_string(),
        syn::Member::Unnamed(idx) => {
            return Err(format!(
                "tuple field access `.{}` is not supported in shaders",
                idx.index
            ));
        }
    };
    if !is_swizzle(&member) {
        return Err(format!(
            "unsupported field/swizzle `.{member}` in shader body"
        ));
    }
    Ok(format!("{base}.{member}"))
}

/// Whether `s` is a valid vector swizzle over {x,y,z,w} or {r,g,b,a}.
fn is_swizzle(s: &str) -> bool {
    if s.is_empty() || s.len() > 4 {
        return false;
    }
    let xyzw = s.chars().all(|c| matches!(c, 'x' | 'y' | 'z' | 'w'));
    let rgba = s.chars().all(|c| matches!(c, 'r' | 'g' | 'b' | 'a'));
    xyzw || rgba
}

/// A method call. The DSL surface has no user methods; the only method-shaped
/// construct is a swizzle mistakenly written as a call, so this rejects with a
/// clear message rather than emitting an undefined MSL method.
fn emit_method_call(m: &syn::ExprMethodCall) -> Result<String, String> {
    Err(format!(
        "method call `.{}(...)` is not supported in shaders (use free intrinsics like `min`, `mix`, `length`)",
        m.method
    ))
}

/// A call expression: a `VecN::new` constructor, a `sample(slot, uv)` texture
/// fetch, or a free intrinsic (`min`, `mix`, `smoothstep`, `length`, …).
fn emit_call(c: &syn::ExprCall) -> Result<String, String> {
    let Expr::Path(callee) = c.func.as_ref() else {
        return Err("unsupported call target in shader body".to_string());
    };
    let path = &callee.path;

    // `Vec4::new(...)` / `Vec3::new(...)` / `Vec2::new(...)` — regardless of how
    // the path was tokenized/wrapped, syn gives us the segments directly.
    if path.segments.len() == 2 && path.segments[1].ident == "new" {
        let ctor = path.segments[0].ident.to_string();
        if let Some(msl_ty) = vec_ctor_msl(&ctor) {
            let args = emit_args(&c.args)?;
            return Ok(format!("{msl_ty}({args})"));
        }
        return Err(format!(
            "unsupported constructor `{ctor}::new` in shader body"
        ));
    }

    // Single-segment call: an intrinsic or `sample`.
    let Some(name) = path.get_ident().map(|i| i.to_string()) else {
        return Err(format!(
            "unsupported call `{}(...)` in shader body",
            path_to_string(path)
        ));
    };

    // `sample(slot, uv)` — the parser keeps the canonical slot form; the MSL
    // shader emitter rewrites it to `tex_N.sample(smp_N, uv)` after this pass,
    // so emit it verbatim here.
    if name == "sample" {
        let args = emit_args(&c.args)?;
        return Ok(format!("sample({args})"));
    }

    if let Some(msl_name) = intrinsic_msl_name(&name) {
        let args = emit_args(&c.args)?;
        return Ok(format!("{msl_name}({args})"));
    }

    Err(format!(
        "unsupported function call `{name}(...)` in shader body (not a known intrinsic)"
    ))
}

fn emit_args(
    args: &syn::punctuated::Punctuated<Expr, syn::token::Comma>,
) -> Result<String, String> {
    let mut parts = Vec::with_capacity(args.len());
    for a in args {
        parts.push(emit_expr(a)?);
    }
    Ok(parts.join(", "))
}

/// Extract a simple identifier from a `let` pattern (`let x` / `let mut x`).
fn pat_ident(pat: &syn::Pat) -> Result<String, String> {
    match pat {
        syn::Pat::Ident(pi) => Ok(pi.ident.to_string()),
        other => Err(format!(
            "unsupported `let` pattern in shader body: {other:?}"
        )),
    }
}

fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn stmt_kind(s: &Stmt) -> &'static str {
    match s {
        Stmt::Local(_) => "let binding",
        Stmt::Item(_) => "item",
        Stmt::Expr(_, _) => "expression",
        Stmt::Macro(_) => "macro invocation",
    }
}

fn expr_kind(e: &Expr) -> &'static str {
    match e {
        Expr::Array(_) => "array",
        Expr::Assign(_) => "assignment",
        Expr::Block(_) => "block",
        Expr::Call(_) => "call",
        Expr::Closure(_) => "closure",
        Expr::ForLoop(_) => "for-loop",
        Expr::If(_) => "if-expression",
        Expr::Index(_) => "index",
        Expr::Loop(_) => "loop",
        Expr::Macro(_) => "macro",
        Expr::Match(_) => "match",
        Expr::MethodCall(_) => "method-call",
        Expr::Reference(_) => "reference",
        Expr::Return(_) => "return",
        Expr::Struct(_) => "struct-literal",
        Expr::While(_) => "while-loop",
        _ => "expression",
    }
}

#[cfg(test)]
#[path = "shader_ast_tests.rs"]
mod tests;
