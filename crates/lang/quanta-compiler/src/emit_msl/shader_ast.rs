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
//! swizzle/field access, arithmetic, comparisons, `if/else`, `let`
//! bindings, bounded `for i in A .. N` loops (compile-time-constant
//! bounds, `uint` counter — a native MSL `for`, the twin of the SPIR-V
//! structured loop), and the shared-struct varying interface: a vertex body
//! ends in the Varyings struct literal (each field lowered to an assignment
//! on the stage-out struct), and a fragment body reads varyings as
//! `<receiver>.<field>` against its stage-in struct. Anything outside that
//! surface is rejected here with a clear `Err(String)` naming the construct —
//! the caller (`shader_pipeline`) turns that into a build error, so an
//! unsupported shape fails loudly rather than miscompiling.

use std::collections::HashMap;
use std::fmt::Write as _;

use syn::{BinOp, Expr, Lit, Stmt, UnOp};

/// The MSL value types a shader expression can have: float scalars/vectors,
/// the `uint` scalar (u32 attributes/varyings and `Nu32` literals), plus
/// `Bool` for comparison results and `Slice` for a `&[T]` storage-buffer
/// array param (whose element type is the boxed inner scalar/vector).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MslType {
    Float,
    /// 32-bit unsigned scalar (`uint`) — a u32 param or `Nu32` literal.
    Uint,
    Vec2,
    Vec3,
    Vec4,
    Bool,
    /// A `&[T]` slice param. The element is one of Float/Vec2/Vec4 (validated at
    /// parse time). Only indexing (`name[i]`) produces a value from it.
    Slice(SliceElem),
    /// Type couldn't be inferred (an unknown identifier, an intrinsic whose
    /// result type we don't model). Only fatal where a concrete declaration is
    /// required (a `let x = if ...` whose value type must be named).
    Unknown,
}

/// The element type of a `&[T]` slice param.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SliceElem {
    Float,
    Vec2,
    Vec4,
}

impl SliceElem {
    /// The MSL type indexing this slice yields.
    fn element(self) -> MslType {
        match self {
            SliceElem::Float => MslType::Float,
            SliceElem::Vec2 => MslType::Vec2,
            SliceElem::Vec4 => MslType::Vec4,
        }
    }
}

impl MslType {
    fn name(self) -> &'static str {
        match self {
            MslType::Float => "float",
            MslType::Uint => "uint",
            MslType::Vec2 => "float2",
            MslType::Vec3 => "float3",
            MslType::Vec4 => "float4",
            MslType::Bool => "bool",
            // A slice is never the declared type of a `let` binding, so this
            // spelling is only a fallback; index it to get a concrete element.
            MslType::Slice(_) => "auto",
            MslType::Unknown => "auto",
        }
    }

    pub(crate) fn from_shader_type(ty: quanta_ir::ShaderType) -> MslType {
        match ty {
            quanta_ir::ShaderType::F32 => MslType::Float,
            quanta_ir::ShaderType::U32 => MslType::Uint,
            quanta_ir::ShaderType::Vec2 => MslType::Vec2,
            quanta_ir::ShaderType::Vec3 => MslType::Vec3,
            quanta_ir::ShaderType::Vec4 => MslType::Vec4,
            // Matrices aren't values a shader body constructs; treat as unknown.
            quanta_ir::ShaderType::Mat4 | quanta_ir::ShaderType::Mat3 => MslType::Unknown,
        }
    }

    /// The slice type for a `&[T]` param whose element `ShaderType` is `ty`.
    pub(crate) fn slice_of(ty: quanta_ir::ShaderType) -> MslType {
        let elem = match ty {
            quanta_ir::ShaderType::F32 => SliceElem::Float,
            quanta_ir::ShaderType::Vec2 => SliceElem::Vec2,
            // Slice element types are validated to f32/Vec2/Vec4 at parse time.
            _ => SliceElem::Vec4,
        };
        MslType::Slice(elem)
    }
}

/// Name → inferred MSL type for every param and in-scope `let` binding.
///
/// A shader body is a flat cascade of `let`s (plus nested branch blocks that
/// only add locals used within the branch), so a single growing map is enough —
/// branch-local names never escape their block, and no shader shadows a name.
type TypeEnv = HashMap<String, MslType>;

/// The walker's scope: the type environment plus the shared-struct varying
/// interface (when the shader uses it). The fragment's receiver param and the
/// vertex's tail struct literal both resolve against `varyings`.
#[derive(Clone)]
pub(crate) struct Scope<'a> {
    env: TypeEnv,
    varyings: Option<&'a quanta_ir::ShaderVaryings>,
}

impl<'a> Scope<'a> {
    /// The fragment receiver param's name, when the varying interface has one.
    fn receiver(&self) -> Option<&str> {
        self.varyings.and_then(|v| v.binding.as_deref())
    }

    /// Whether `name` is the fragment's Varyings receiver.
    fn is_receiver(&self, name: &str) -> bool {
        self.receiver() == Some(name)
    }

    /// The MSL type of `field` on the varyings struct: the position field is
    /// a `float4` ([[position]] — the interpolated window position in a
    /// fragment), a declared varying maps its ShaderType, anything else is
    /// `None`.
    fn varying_field(&self, field: &str) -> Option<MslType> {
        let v = self.varyings?;
        if field == v.position {
            return Some(MslType::Vec4);
        }
        v.field_type(field).map(MslType::from_shader_type)
    }

    fn struct_name(&self) -> &str {
        self.varyings.map(|v| v.struct_name.as_str()).unwrap_or("")
    }
}

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
        // Screen-space derivatives — fragment-stage builtins.
        "fwidth" => "fwidth",
        "dpdx" => "dfdx",
        "dpdy" => "dfdy",
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

/// How the block's tail expression is routed — the public face of the
/// internal [`TailMode`].
pub(crate) enum BodyTail<'a> {
    /// Emit `<var> = <tail>;` (vertex position-only: the result feeds an
    /// output struct member).
    Assign(&'a str),
    /// Emit `return <tail>;` (fragment: the tail is the output color).
    Return,
    /// The tail must be the Varyings struct literal; each field lowers to
    /// `<out_var>.<field> = <expr>;` (vertex shared-struct model).
    StructOut(&'a str),
}

/// Emit the MSL body of a shader block.
///
/// `tail` routes the block's final expression (see [`BodyTail`]); `varyings`
/// carries the shared-struct interface when the shader uses it. Assign/Return
/// tails handle a tail that is itself an `if/else` by lowering the branch
/// arms to assignments/returns; a StructOut tail requires the literal itself.
pub(crate) fn emit_body(
    body_source: &str,
    tail: BodyTail,
    params: &[(String, MslType)],
    varyings: Option<&quanta_ir::ShaderVaryings>,
) -> Result<String, String> {
    let block: syn::Block = syn::parse_str(body_source)
        .map_err(|e| format!("failed to parse shader body as a Rust block: {e}"))?;
    let mut scope = Scope {
        env: params.iter().cloned().collect(),
        varyings,
    };
    let mode = match tail {
        BodyTail::Assign(var) => TailMode::Route(Some(var)),
        BodyTail::Return => TailMode::Route(None),
        BodyTail::StructOut(var) => TailMode::StructOut(var),
    };
    let mut out = String::new();
    emit_stmts(&block.stmts, mode, 1, &mut scope, &mut out)?;
    Ok(out)
}

/// Emit a statement sequence. The final expression statement (no trailing `;`)
/// is the block's value and is routed through `tail`.
fn emit_stmts(
    stmts: &[Stmt],
    mode: TailMode,
    indent: usize,
    scope: &mut Scope,
    out: &mut String,
) -> Result<(), String> {
    for (i, stmt) in stmts.iter().enumerate() {
        let is_last = i == stmts.len() - 1;
        match stmt {
            Stmt::Local(local) => emit_local(local, indent, scope, out)?,
            Stmt::Expr(expr, semi) => {
                if is_last && semi.is_none() {
                    emit_tail(expr, mode, indent, scope, out)?;
                } else {
                    match expr {
                        // Statement-position `if`: no value, branches hold
                        // assignments/lets (the branch-and-assign shape).
                        Expr::If(if_expr) => {
                            emit_if(if_expr, TailMode::Statement, indent, scope, out)?
                        }
                        // `name = expr;` — assignment to a `let mut` local.
                        Expr::Assign(assign) => emit_assign(assign, indent, scope, out)?,
                        // `for i in A .. N { … }` — a bounded counted loop
                        // (block-like, so syn records no trailing semi).
                        Expr::ForLoop(for_loop) => emit_for(for_loop, indent, scope, out)?,
                        // Any other non-tail expression statement has no
                        // effect in a shader body.
                        other => {
                            return Err(format!(
                                "expression statement with unused value in shader body: {}",
                                expr_kind(other)
                            ));
                        }
                    }
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

/// Emit `name = expr;` — the target must be a plain identifier already
/// bound by a `let mut`.
fn emit_assign(
    assign: &syn::ExprAssign,
    indent: usize,
    scope: &mut Scope,
    out: &mut String,
) -> Result<(), String> {
    let pad = "    ".repeat(indent);
    let name = match assign.left.as_ref() {
        Expr::Path(path) => path
            .path
            .get_ident()
            .map(|i| i.to_string())
            .ok_or_else(|| "assignment target must be a plain identifier".to_string())?,
        other => {
            return Err(format!(
                "unsupported assignment target in shader body: {}",
                expr_kind(other)
            ));
        }
    };
    if !scope.env.contains_key(&name) {
        return Err(format!("assignment to unknown local `{name}`"));
    }
    let ty = infer_type(&assign.right, scope);
    let code = emit_expr(&assign.right, scope)?;
    scope.env.insert(name.clone(), ty);
    writeln!(out, "{pad}{name} = {code};").unwrap();
    Ok(())
}

/// Emit `for i in A .. N { … }` as a native MSL `for` over a `uint` counter.
///
/// The bounds must be compile-time integer literals (the SPIR-V twin enforces
/// the same — an unbounded shader loop is invalid), the range half-open
/// ascending (`..=` rejected), and the counter a fresh name (the SPIR-V
/// parser's scope rule; enforcing it here keeps the accept/reject verdicts in
/// lockstep). The body is a statement block: assignments to outer `let mut`
/// locals, nested `if`s/loops, branch-local `let`s — no value tail.
fn emit_for(
    for_loop: &syn::ExprForLoop,
    indent: usize,
    scope: &mut Scope,
    out: &mut String,
) -> Result<(), String> {
    let pad = "    ".repeat(indent);
    if for_loop.label.is_some() {
        return Err("labeled `for` loops are not supported in shaders".to_string());
    }
    let counter = match for_loop.pat.as_ref() {
        syn::Pat::Ident(pi) => pi.ident.to_string(),
        _ => return Err("the for-loop counter must be a plain identifier".to_string()),
    };
    if scope.env.contains_key(&counter) {
        return Err(format!(
            "for-loop counter `{counter}` shadows an existing binding"
        ));
    }
    let Expr::Range(range) = for_loop.expr.as_ref() else {
        return Err("for-loop iterable must be a literal range `A .. N`".to_string());
    };
    if matches!(range.limits, syn::RangeLimits::Closed(_)) {
        return Err(
            "inclusive `..=` ranges are not supported in shader for-loops (use `A .. N`)"
                .to_string(),
        );
    }
    let start = const_loop_bound(range.start.as_deref())?;
    let end = const_loop_bound(range.end.as_deref())?;
    writeln!(
        out,
        "{pad}for (uint {counter} = {start}u; {counter} < {end}u; ++{counter}) {{"
    )
    .unwrap();
    scope.env.insert(counter.clone(), MslType::Uint);
    emit_stmts(
        &for_loop.body.stmts,
        TailMode::Statement,
        indent + 1,
        scope,
        out,
    )?;
    // The counter is scoped to the loop (the MSL for-init declares it);
    // dropping it from the env keeps post-loop uses a named error, matching
    // the SPIR-V parser popping its counter local.
    scope.env.remove(&counter);
    writeln!(out, "{pad}}}").unwrap();
    Ok(())
}

/// Extract a compile-time u32 loop bound from a range endpoint: an integer
/// literal, bare (`8`) or `u32`-suffixed (`8u32`). Anything else — a param, a
/// local, an expression — is rejected so an unbounded shader loop can never
/// be emitted. (Const-expression bounds like `2 * 4` are deferred.)
fn const_loop_bound(endpoint: Option<&Expr>) -> Result<u32, String> {
    let Some(mut e) = endpoint else {
        return Err("for-loop range must spell both bounds (`A .. N`)".to_string());
    };
    loop {
        match e {
            Expr::Paren(p) => e = &p.expr,
            Expr::Group(g) => e = &g.expr,
            _ => break,
        }
    }
    if let Expr::Lit(l) = e
        && let Lit::Int(i) = &l.lit
        && (i.suffix().is_empty() || i.suffix() == "u32")
        && let Ok(v) = i.base10_parse::<u32>()
    {
        return Ok(v);
    }
    Err("for-loop bound must be a compile-time constant integer literal (`A .. N`)".to_string())
}

/// Emit the block's tail expression, routed to a return or an assignment.
///
/// A tail `if/else` is lowered structurally under the Assign/Route modes:
/// each arm emits its own return/assignment, so no MSL ternary or phi temp is
/// needed. Under StructOut the tail MUST be the Varyings struct literal.
fn emit_tail(
    expr: &Expr,
    mode: TailMode,
    indent: usize,
    scope: &mut Scope,
    out: &mut String,
) -> Result<(), String> {
    let pad = "    ".repeat(indent);
    if let TailMode::StructOut(out_var) = mode {
        let Expr::Struct(st) = expr else {
            return Err(format!(
                "the vertex body must end in a `{} {{ .. }}` struct literal \
                 (the Varyings interface is built exactly once, in tail position)",
                scope.struct_name()
            ));
        };
        return emit_struct_out(st, out_var, indent, scope, out);
    }
    if let Expr::If(if_expr) = expr {
        // The nested if inherits the mode: value-routing stays
        // value-routing, statement position stays statement position.
        emit_if(if_expr, mode, indent, scope, out)?;
        return Ok(());
    }
    if let Expr::Assign(assign) = expr {
        // A trailing assignment (no `;` on the last statement) still
        // yields no value — treat as a statement.
        return emit_assign(assign, indent, scope, out);
    }
    if let Expr::ForLoop(for_loop) = expr {
        // A `for` yields no value: fine as the last statement of a
        // statement-position branch, never as the body's result value.
        if matches!(mode, TailMode::Statement) {
            return emit_for(for_loop, indent, scope, out);
        }
        return Err(
            "a `for` loop yields no value; the shader body must end in a value expression"
                .to_string(),
        );
    }
    let code = emit_expr(expr, scope)?;
    match mode {
        TailMode::Assign(var) | TailMode::Route(Some(var)) => {
            writeln!(out, "{pad}{var} = {code};").unwrap()
        }
        TailMode::Route(None) => writeln!(out, "{pad}return {code};").unwrap(),
        TailMode::Statement => {
            return Err(
                "a statement-position `if` branch cannot end with a value expression".to_string(),
            );
        }
        // Handled above.
        TailMode::StructOut(_) => unreachable!(),
    }
    Ok(())
}

/// Lower the vertex tail's Varyings struct literal: every declared field —
/// the `#[position]` field and each varying — must appear exactly once (any
/// order; field-init shorthand allowed; `..rest` rejected), and each value
/// assigns to the matching member of the stage-out struct. Where the walker
/// can infer a field value's type, a mismatch against the declared field type
/// is rejected (the SPIR-V twin enforces the same, strictly).
fn emit_struct_out(
    st: &syn::ExprStruct,
    out_var: &str,
    indent: usize,
    scope: &mut Scope,
    out: &mut String,
) -> Result<(), String> {
    let pad = "    ".repeat(indent);
    let varyings = scope
        .varyings
        .ok_or_else(|| "internal: StructOut tail without a varyings interface".to_string())?;
    let lit_name = st
        .path
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();
    if lit_name != varyings.struct_name {
        return Err(format!(
            "the vertex body must end in a `{} {{ .. }}` struct literal, found `{lit_name}`",
            varyings.struct_name
        ));
    }
    if st.rest.is_some() {
        return Err(format!(
            "`..` struct-update syntax is not supported in the `{} {{ .. }}` literal",
            varyings.struct_name
        ));
    }

    // Collect `(field, expr)` in written order, checking duplicates/unknowns.
    let mut seen: Vec<(String, &Expr)> = Vec::new();
    for fv in &st.fields {
        let syn::Member::Named(id) = &fv.member else {
            return Err(format!(
                "tuple-style fields are not supported in the `{} {{ .. }}` literal",
                varyings.struct_name
            ));
        };
        let fname = id.to_string();
        if fname != varyings.position && varyings.field_type(&fname).is_none() {
            return Err(format!(
                "unknown field `{fname}` in the `{} {{ .. }}` literal",
                varyings.struct_name
            ));
        }
        if seen.iter().any(|(n, _)| *n == fname) {
            return Err(format!(
                "duplicate field `{fname}` in the `{} {{ .. }}` literal",
                varyings.struct_name
            ));
        }
        seen.push((fname, &fv.expr));
    }

    // Emit assignments in declaration order: position first, then fields.
    let mut ordered: Vec<(&str, MslType)> = Vec::with_capacity(1 + varyings.fields.len());
    ordered.push((varyings.position.as_str(), MslType::Vec4));
    for f in &varyings.fields {
        ordered.push((f.name.as_str(), MslType::from_shader_type(f.ty)));
    }
    for (name, declared) in ordered {
        let Some((_, expr)) = seen.iter().find(|(n, _)| n == name) else {
            return Err(format!(
                "missing field `{name}` in the `{} {{ .. }}` literal",
                varyings.struct_name
            ));
        };
        let inferred = infer_type(expr, scope);
        // Scalar fields tolerate the f32/uint pair (MSL's usual arithmetic
        // conversions; the SPIR-V twin converts explicitly). Anything else
        // that infers to a KNOWN type must match the declaration.
        let scalar_pair = matches!(
            (inferred, declared),
            (MslType::Float, MslType::Uint) | (MslType::Uint, MslType::Float)
        );
        if inferred != MslType::Unknown && inferred != declared && !scalar_pair {
            return Err(format!(
                "field `{name}` of `{}` expects {} but the literal provides {}",
                varyings.struct_name,
                declared.name(),
                inferred.name()
            ));
        }
        let code = emit_expr(expr, scope)?;
        writeln!(out, "{pad}{out_var}.{name} = {code};").unwrap();
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
    /// The block tail must be the Varyings struct literal, lowered to member
    /// assignments on the named stage-out struct (vertex shared-struct
    /// model). Never inherited by `if` branches — a literal inside a branch
    /// is rejected.
    StructOut(&'a str),
    /// Statement position: the `if` yields no value; branches contain
    /// statements (assignments to `let mut` locals) only.
    Statement,
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
    scope: &mut Scope,
    out: &mut String,
) -> Result<(), String> {
    let pad = "    ".repeat(indent);
    let cond = emit_expr(&if_expr.cond, scope)?;
    writeln!(out, "{pad}if ({cond}) {{").unwrap();
    emit_branch_block(&if_expr.then_branch.stmts, mode, indent + 1, scope, out)?;

    match &if_expr.else_branch {
        Some((_, else_expr)) => {
            writeln!(out, "{pad}}} else {{").unwrap();
            match else_expr.as_ref() {
                // `else if` chains as a nested if inside the else block.
                Expr::If(nested) => emit_if(nested, mode, indent + 1, scope, out)?,
                Expr::Block(block) => {
                    emit_branch_block(&block.block.stmts, mode, indent + 1, scope, out)?
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
    scope: &mut Scope,
    out: &mut String,
) -> Result<(), String> {
    emit_stmts(stmts, mode, indent, scope, out)
}

/// Infer the MSL type an `if/else` evaluates to, from its then-branch tail.
///
/// The branch may declare its own `let`s before the tail (`{ let d = ..; d }`),
/// so inference threads a scratch env seeded from the outer scope through those
/// bindings before typing the tail — otherwise a tail that names a branch-local
/// would read as `Unknown`.
fn infer_if_type(if_expr: &syn::ExprIf, scope: &Scope) -> MslType {
    infer_branch_type(&if_expr.then_branch.stmts, scope)
}

/// Infer the value type of a branch block by simulating its `let` bindings.
fn infer_branch_type(stmts: &[Stmt], outer: &Scope) -> MslType {
    let mut scratch = outer.clone();
    for (i, stmt) in stmts.iter().enumerate() {
        match stmt {
            Stmt::Local(local) => {
                if let (Ok(name), Some(init)) = (pat_ident(&local.pat), local.init.as_ref()) {
                    let ty = infer_type(&init.expr, &scratch);
                    scratch.env.insert(name, ty);
                }
            }
            Stmt::Expr(expr, None) if i == stmts.len() - 1 => {
                return infer_type(expr, &scratch);
            }
            _ => {}
        }
    }
    MslType::Unknown
}

/// Infer the MSL type of an expression from structure. Unmodeled shapes return
/// `Unknown`; that's only fatal where a concrete type must be named (if-let).
fn infer_type(expr: &Expr, scope: &Scope) -> MslType {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Bool(_) => MslType::Bool,
            // `3u32` is the DSL's explicit unsigned literal; a bare `3` stays
            // float (backward compatibility — see `emit_lit`).
            Lit::Int(i) if i.suffix() == "u32" => MslType::Uint,
            _ => MslType::Float,
        },
        Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|id| scope.env.get(&id.to_string()).copied())
            .unwrap_or(MslType::Unknown),
        Expr::Paren(p) => infer_type(&p.expr, scope),
        Expr::Group(g) => infer_type(&g.expr, scope),
        Expr::Cast(c) => infer_type(&c.expr, scope),
        Expr::Unary(u) => match u.op {
            UnOp::Not(_) => MslType::Bool,
            // Deref of a `&Vec2` uniform yields the element type; but we only
            // reach here for `*name`, whose element type is the param's type.
            _ => infer_type(&u.expr, scope),
        },
        Expr::Binary(b) => infer_binary_type(b, scope),
        Expr::Field(f) => infer_field_type(f, scope),
        Expr::If(if_expr) => infer_if_type(if_expr, scope),
        Expr::Call(c) => infer_call_type(c, scope),
        // `slice[i]` — the element type of a `&[T]` slice param.
        Expr::Index(idx) => match idx.expr.as_ref() {
            Expr::Path(path) => path
                .path
                .get_ident()
                .and_then(|id| scope.env.get(&id.to_string()).copied())
                .map(|t| match t {
                    MslType::Slice(elem) => elem.element(),
                    _ => MslType::Unknown,
                })
                .unwrap_or(MslType::Unknown),
            _ => MslType::Unknown,
        },
        _ => MslType::Unknown,
    }
}

fn infer_binary_type(b: &syn::ExprBinary, scope: &Scope) -> MslType {
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
            let l = infer_type(&b.left, scope);
            let r = infer_type(&b.right, scope);
            match (l, r) {
                (MslType::Uint, MslType::Uint) => MslType::Uint,
                // Mixed uint/float arithmetic is float — MSL's usual
                // arithmetic conversions promote the uint operand, and the
                // SPIR-V side widens it with OpConvertUToF (emit_arith_op).
                (MslType::Uint, MslType::Float) | (MslType::Float, MslType::Uint) => MslType::Float,
                (MslType::Float, other) | (MslType::Unknown, other) if other != MslType::Float => {
                    other
                }
                (l, _) => l,
            }
        }
    }
}

/// Field result type: a receiver field reads its DECLARED type from the
/// varyings interface; otherwise the swizzle heuristic (single component →
/// float, N-char swizzle → floatN).
fn infer_field_type(f: &syn::ExprField, scope: &Scope) -> MslType {
    if let syn::Member::Named(id) = &f.member {
        // `s.<field>` — the varyings receiver's field carries its declared
        // type (`s.kind` is uint, `s.<position>` float4).
        if let Expr::Path(p) = f.base.as_ref()
            && let Some(base) = p.path.get_ident()
            && scope.is_receiver(&base.to_string())
        {
            return scope
                .varying_field(&id.to_string())
                .unwrap_or(MslType::Unknown);
        }
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

fn infer_call_type(c: &syn::ExprCall, scope: &Scope) -> MslType {
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
        // `sample()` returns a full Vec4; so does `frag_coord()` (xyzw =
        // window x, window y, depth, 1/w).
        "sample" | "frag_coord" => MslType::Vec4,
        // The vertex-index builtins are uint — comparisons against them stay
        // integer (`== 1u`), mirroring the SPIR-V unsigned opcode family.
        "vertex_id" | "instance_id" => MslType::Uint,
        // Reductions to a scalar.
        "length" | "distance" | "dot" => MslType::Float,
        // Component-wise intrinsics take the type of their first argument.
        "min" | "max" | "clamp" | "mix" | "abs" | "floor" | "ceil" | "round" | "fract"
        | "normalize" | "step" | "smoothstep" | "pow" | "sin" | "cos" | "tan" | "sqrt" | "exp"
        | "log" | "exp2" | "log2" | "fma" | "cross" => c
            .args
            .first()
            .map(|a| infer_type(a, scope))
            .unwrap_or(MslType::Unknown),
        _ => MslType::Unknown,
    }
}

/// Emit an expression to an MSL source fragment (no trailing `;`).
///
/// `scope` is threaded so that a `name[index]` on a slice param can be
/// validated (only `&[T]` params index) and so receiver-field access resolves
/// against the varyings interface — every other expression is spacing-/
/// type-blind.
fn emit_expr(expr: &Expr, scope: &Scope) -> Result<String, String> {
    match expr {
        Expr::Lit(lit) => emit_lit(&lit.lit),
        Expr::Path(path) => emit_path_ident(&path.path, scope),
        Expr::Paren(p) => Ok(format!("({})", emit_expr(&p.expr, scope)?)),
        Expr::Group(g) => emit_expr(&g.expr, scope),
        Expr::Unary(u) => emit_unary(u, scope),
        Expr::Binary(b) => emit_binary(b, scope),
        // `x as f32` / `as u32` etc. — shaders are float-only, strip the cast.
        Expr::Cast(c) => emit_expr(&c.expr, scope),
        Expr::Field(f) => emit_field(f, scope),
        Expr::Index(idx) => emit_index(idx, scope),
        Expr::MethodCall(m) => emit_method_call(m),
        Expr::Call(c) => emit_call(c, scope),
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

/// Index a `&[T]` slice param: `name[index]` → `name[(uint)(index)]`. The index
/// truncates to `uint` (matching the SPIR-V `OpConvertFToU`); bounds are
/// UNCHECKED (the GPU storage-buffer contract). Indexing a non-slice value or
/// an unknown name is a named error.
fn emit_index(idx: &syn::ExprIndex, scope: &Scope) -> Result<String, String> {
    let base = match idx.expr.as_ref() {
        Expr::Path(path) => path
            .path
            .get_ident()
            .map(|i| i.to_string())
            .ok_or_else(|| "slice index base must be a plain identifier".to_string())?,
        other => {
            return Err(format!(
                "cannot index {} in shader body; only `&[T]` slice params support indexing",
                expr_kind(other)
            ));
        }
    };
    if !matches!(scope.env.get(&base), Some(MslType::Slice(_))) {
        return Err(format!(
            "`{base}[..]` indexes a non-slice value; only `&[T]` slice params support indexing"
        ));
    }
    let index = emit_expr(&idx.index, scope)?;
    Ok(format!("{base}[(uint)({index})]"))
}

fn emit_lit(lit: &Lit) -> Result<String, String> {
    match lit {
        Lit::Float(f) => {
            // Strip any Rust suffix (`1.0f32`) and guarantee a decimal point so
            // MSL types the literal as float, not int.
            let s = f.base10_digits();
            Ok(ensure_float_literal(s))
        }
        // `3u32` spells as the MSL `uint` literal `3u`; a BARE integer keeps
        // the historical float spelling (`3` → `3.0`) so existing float
        // bodies are untouched — a bare literal compared against a `uint`
        // gets the integer spelling in `emit_binary` instead.
        Lit::Int(i) if i.suffix() == "u32" => Ok(format!("{}u", i.base10_digits())),
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
/// constructor callee (handled in `emit_call`), never as a bare value. The
/// varyings receiver is NOT a value — only its fields are.
fn emit_path_ident(path: &syn::Path, scope: &Scope) -> Result<String, String> {
    if let Some(ident) = path.get_ident() {
        let name = ident.to_string();
        if scope.is_receiver(&name) {
            return Err(format!(
                "the varyings struct `{name}` can only be read by field access \
                 (`{name}.<field>`)"
            ));
        }
        return Ok(name);
    }
    Err(format!(
        "unsupported path expression in shader body: `{}`",
        path_to_string(path)
    ))
}

fn emit_unary(u: &syn::ExprUnary, scope: &Scope) -> Result<String, String> {
    let inner = emit_expr(&u.expr, scope)?;
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

/// Emit a comparison operand whose OTHER side is `uint`-typed: a bare
/// (unsuffixed) integer literal spells as a `uint` literal (`3` → `3u`) so
/// the MSL comparison stays integer — mirroring the SPIR-V side, which
/// coerces the literal with `OpConvertFToU` and compares with the unsigned
/// opcode family. Anything else emits normally (MSL's arithmetic conversions
/// cover the remaining mixed cases).
fn emit_cmp_operand_against_uint(expr: &Expr, scope: &Scope) -> Result<String, String> {
    if let Expr::Lit(lit) = expr
        && let Lit::Int(i) = &lit.lit
        && i.suffix().is_empty()
    {
        return Ok(format!("{}u", i.base10_digits()));
    }
    emit_expr(expr, scope)
}

fn emit_binary(b: &syn::ExprBinary, scope: &Scope) -> Result<String, String> {
    let is_cmp = matches!(
        b.op,
        BinOp::Lt(_) | BinOp::Gt(_) | BinOp::Le(_) | BinOp::Ge(_) | BinOp::Eq(_) | BinOp::Ne(_)
    );
    let (l, r) = if is_cmp {
        // A comparison with a uint side keeps the whole comparison integer:
        // the opposite bare int literal spells as `Nu` (see the helper).
        let lt = infer_type(&b.left, scope);
        let rt = infer_type(&b.right, scope);
        let l = if rt == MslType::Uint {
            emit_cmp_operand_against_uint(&b.left, scope)?
        } else {
            emit_expr(&b.left, scope)?
        };
        let r = if lt == MslType::Uint {
            emit_cmp_operand_against_uint(&b.right, scope)?
        } else {
            emit_expr(&b.right, scope)?
        };
        (l, r)
    } else {
        (emit_expr(&b.left, scope)?, emit_expr(&b.right, scope)?)
    };
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

/// Field / swizzle access. A receiver field (`s.uv`, `s.<position>`) passes
/// through as a struct-member read validated against the varyings interface;
/// `.x/.y/.z/.w` and `.r/.g/.b/.a` map straight through (MSL uses the same
/// swizzle spellings); multi-char swizzles (`.xy`, `.zw`) are supported too
/// since the runtime shaders use `uv_rect.zw`.
fn emit_field(f: &syn::ExprField, scope: &Scope) -> Result<String, String> {
    let member = match &f.member {
        syn::Member::Named(id) => id.to_string(),
        syn::Member::Unnamed(idx) => {
            return Err(format!(
                "tuple field access `.{}` is not supported in shaders",
                idx.index
            ));
        }
    };
    // `s.<field>` — the varyings receiver reads a declared field by name.
    if let Expr::Path(p) = f.base.as_ref()
        && let Some(base) = p.path.get_ident()
    {
        let base = base.to_string();
        if scope.is_receiver(&base) {
            if scope.varying_field(&member).is_some() {
                return Ok(format!("{base}.{member}"));
            }
            return Err(format!(
                "no field `{member}` on the varyings struct `{}`",
                scope.struct_name()
            ));
        }
    }
    let base = emit_expr(&f.base, scope)?;
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
fn emit_call(c: &syn::ExprCall, scope: &Scope) -> Result<String, String> {
    let Expr::Path(callee) = c.func.as_ref() else {
        return Err("unsupported call target in shader body".to_string());
    };
    let path = &callee.path;

    // `Vec4::new(...)` / `Vec3::new(...)` / `Vec2::new(...)` — regardless of how
    // the path was tokenized/wrapped, syn gives us the segments directly.
    if path.segments.len() == 2 && path.segments[1].ident == "new" {
        let ctor = path.segments[0].ident.to_string();
        if let Some(msl_ty) = vec_ctor_msl(&ctor) {
            let args = emit_args(&c.args, scope)?;
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
        let args = emit_args(&c.args, scope)?;
        return Ok(format!("sample({args})"));
    }

    // `frag_coord()` — the window-space position builtin. The fragment
    // emitter declares a `float4 _frag_coord [[position]]` parameter whenever
    // the body calls this, so the call lowers directly to that identifier
    // (unlike `sample`, no post-pass rewrite is needed: the parameter name is
    // fixed and the call is argument-free). Vertex bodies never reach here —
    // `emit_vertex_shader` rejects `frag_coord()` before body lowering.
    if name == "frag_coord" {
        if !c.args.is_empty() {
            return Err("frag_coord() takes no arguments".to_string());
        }
        return Ok("_frag_coord".to_string());
    }

    // `vertex_id()` / `instance_id()` — the vertex-index builtins. The
    // vertex emitter declares `uint _vertex_id [[vertex_id]]` /
    // `uint _instance_id [[instance_id]]` parameters whenever the body calls
    // them, so each call lowers directly to its identifier (argument-free
    // static-name builtins, like `frag_coord` — no post-pass rewrite).
    // Fragment bodies never reach here — `emit_fragment_shader` rejects both
    // before body lowering.
    if name == "vertex_id" || name == "instance_id" {
        if !c.args.is_empty() {
            return Err(format!("{name}() takes no arguments"));
        }
        return Ok(format!("_{name}"));
    }

    if let Some(msl_name) = intrinsic_msl_name(&name) {
        let args = emit_args(&c.args, scope)?;
        return Ok(format!("{msl_name}({args})"));
    }

    Err(format!(
        "unsupported function call `{name}(...)` in shader body (not a known intrinsic)"
    ))
}

fn emit_args(
    args: &syn::punctuated::Punctuated<Expr, syn::token::Comma>,
    scope: &Scope,
) -> Result<String, String> {
    let mut parts = Vec::with_capacity(args.len());
    for a in args {
        parts.push(emit_expr(a, scope)?);
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
    scope: &mut Scope,
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
        let ty = infer_if_type(if_expr, scope);
        if ty == MslType::Unknown {
            return Err(format!(
                "cannot infer the type of `let {name} = if ...` — the branch value type is needed to declare the MSL local"
            ));
        }
        scope.env.insert(name.clone(), ty);
        writeln!(out, "{pad}{} {name};", ty.name()).unwrap();
        emit_if(if_expr, TailMode::Assign(&name), indent, scope, out)?;
    } else {
        let ty = infer_type(&init.expr, scope);
        let code = emit_expr(&init.expr, scope)?;
        scope.env.insert(name.clone(), ty);
        // `auto` is fine for an initialized binding; only if-lowered locals need
        // an explicit type. Keep `auto` to stay agnostic where inference is
        // unsure but the initializer makes it concrete for the MSL compiler.
        writeln!(out, "{pad}auto {name} = {code};").unwrap();
    }
    Ok(())
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
