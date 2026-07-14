//! Recursive-descent WGSL walker for render-shader bodies.
//!
//! The SPIR-V shader walker (`quanta-compiler/src/emit_spirv/{expr,expr_atom}`)
//! emits SSA instructions; this one emits WGSL *text*, because WGSL is a
//! friendlier target than SPIR-V or MSL — statement-`if`/`else` and the value
//! form `let x = if a { b } else { c }` are native (no `OpPhi` construction),
//! and almost every intrinsic keeps its Rust name. The construct surface is a
//! mirror of the SPIR-V walker: what SPIR-V accepts, this accepts; what SPIR-V
//! rejects (a method call, a `for`/`while` loop, an unknown intrinsic, an
//! `if` without `else` as an expression, an out-of-range swizzle, indexing a
//! non-slice), this rejects with a clear `Err`.
//!
//! The walker threads a `Ctx` of param/slice/local type info. Statements are
//! written into `out`; the final result expression is RETURNED as a string so
//! the caller (`shader.rs`) can place it into a vertex `pos_result` or a
//! fragment `return`. Types (`WType`) are tracked only where the grammar needs
//! them: swizzle arity validation and choosing the result type of a value-`if`.

use super::shader_tokenizer::{ShaderToken, tokenize_shader_expr};
use crate::ShaderType;

/// The value type of a sub-expression. Only the distinctions the walker acts on
/// are kept: scalar vs. the three vector arities vs. the two matrices. Every
/// literal and arithmetic result the grammar produces is one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WType {
    F32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
    Mat3,
}

impl WType {
    fn from_shader(ty: ShaderType) -> WType {
        match ty {
            ShaderType::F32 => WType::F32,
            ShaderType::Vec2 => WType::Vec2,
            ShaderType::Vec3 => WType::Vec3,
            ShaderType::Vec4 => WType::Vec4,
            ShaderType::Mat4 => WType::Mat4,
            ShaderType::Mat3 => WType::Mat3,
        }
    }

    /// Component count for a vector type; `None` for scalars/matrices (which
    /// cannot be swizzled). Mirrors the SPIR-V walker's `vector_arity`.
    fn vector_arity(self) -> Option<u32> {
        match self {
            WType::Vec2 => Some(2),
            WType::Vec3 => Some(3),
            WType::Vec4 => Some(4),
            _ => None,
        }
    }
}

/// Per-body context: the params (name → type, whether uniform/slice), the
/// running list of local bindings the statement walker has introduced, and a
/// shared monotonic counter for value-`if` temporaries. The walker never
/// mutates params; `locals` grows as `let`s are seen and shrinks only
/// implicitly (branch bodies clone-and-discard). `next_tmp` is a `Cell` shared
/// by every cloned branch context so a temporary name is unique across the
/// WHOLE body — WGSL has no shadowing across nested `var` declarations, so
/// `_wif0` must never be reused, even in a sibling branch.
struct Ctx<'a> {
    params: &'a [ParamInfo],
    locals: Vec<(String, WType)>,
    next_tmp: &'a core::cell::Cell<u32>,
}

/// One shader param, distilled to what the walker needs. A `&T` uniform needs
/// no flag here — in the body it reads like any other value (the binding is
/// emitted in `shader.rs`); only slices are special (they can be indexed).
pub(super) struct ParamInfo {
    pub name: String,
    pub ty: WType,
    pub is_slice: bool,
}

impl<'a> Ctx<'a> {
    /// Resolve a bare identifier to its type: a local shadows a param of the
    /// same name (Rust scoping), then params, then the two boolean literals.
    fn lookup(&self, name: &str) -> Option<WType> {
        if let Some((_, ty)) = self.locals.iter().rev().find(|(n, _)| n == name) {
            return Some(*ty);
        }
        if let Some(p) = self.params.iter().find(|p| p.name == name) {
            return Some(p.ty);
        }
        None
    }

    fn slice(&self, name: &str) -> Option<&ParamInfo> {
        self.params.iter().find(|p| p.name == name && p.is_slice)
    }
}

/// Convert the DSL params into the walker's `ParamInfo` view.
pub(super) fn param_infos(params: &[crate::ShaderParam]) -> Vec<ParamInfo> {
    params
        .iter()
        .map(|p| ParamInfo {
            name: p.name.clone(),
            ty: WType::from_shader(p.ty),
            is_slice: p.is_slice,
        })
        .collect()
}

/// Walk a shader body into WGSL. `pad` is the per-line indent; every statement
/// is written to `out` with that indent, and the trailing result expression is
/// returned as a `(text, type)` pair. An empty body (no trailing expression) is
/// an error — every shader body ends in a value (the position or the color).
pub(super) fn walk_body(
    body_source: &str,
    params: &[ParamInfo],
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    let src = body_source.trim();
    let src = if src.starts_with('{') && src.ends_with('}') {
        &src[1..src.len() - 1]
    } else {
        src
    };
    let tokens = tokenize_shader_expr(src);
    let next_tmp = core::cell::Cell::new(0u32);
    let mut ctx = Ctx {
        params,
        locals: Vec::new(),
        next_tmp: &next_tmp,
    };
    let mut pos = 0;
    match walk_statements(&tokens, &mut pos, &mut ctx, pad, out)? {
        Some(v) => Ok(v),
        None => Err("shader body has no trailing result expression".to_string()),
    }
}

/// Statement walker: `let [mut]` bindings, assignments to existing locals,
/// statement-`if`/`else`, and a trailing result expression. Each statement is
/// emitted to `out`; a trailing expression (no `;`) becomes the returned value.
/// A `let mut` becomes a WGSL `var`; a plain `let` stays `let`.
fn walk_statements(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<Option<(String, WType)>, String> {
    let mut last: Option<(String, WType)> = None;
    while *pos < tokens.len() {
        // `let [mut] name = expr ;`
        if tokens[*pos] == ShaderToken::Ident("let".to_string()) {
            *pos += 1;
            let is_mut = tokens.get(*pos) == Some(&ShaderToken::Ident("mut".to_string()));
            if is_mut {
                *pos += 1;
            }
            let name = match tokens.get(*pos) {
                Some(ShaderToken::Ident(n)) => n.clone(),
                _ => return Err("expected identifier after `let`".to_string()),
            };
            *pos += 1;
            if tokens.get(*pos) != Some(&ShaderToken::Eq) {
                return Err("expected `=` in let binding".to_string());
            }
            *pos += 1;
            let (expr, ty) = walk_conditional(tokens, pos, ctx, pad, out)?;
            if tokens.get(*pos) == Some(&ShaderToken::Semi) {
                *pos += 1;
            }
            let kw = if is_mut { "var" } else { "let" };
            out.push_str(&format!("{pad}{kw} {name} = {expr};\n"));
            ctx.locals.push((name, ty));
            last = None;
            continue;
        }
        // `name = expr ;` — assignment to an existing local (the `let mut` var).
        if let Some(ShaderToken::Ident(name)) = tokens.get(*pos)
            && tokens.get(*pos + 1) == Some(&ShaderToken::Eq)
        {
            if !ctx.locals.iter().any(|(n, _)| n == name) {
                return Err(format!("assignment to unknown local `{name}`"));
            }
            let name = name.clone();
            *pos += 2;
            let (expr, _) = walk_conditional(tokens, pos, ctx, pad, out)?;
            if tokens.get(*pos) == Some(&ShaderToken::Semi) {
                *pos += 1;
            }
            out.push_str(&format!("{pad}{name} = {expr};\n"));
            last = None;
            continue;
        }
        // Statement-level `if` (emits a WGSL `if`/`else` block; may reassign
        // outer locals in its branches). It can still YIELD a value when both
        // branches end in a trailing expression — the `... } else { color }`
        // tail shape (TailMode::Route) — which becomes the body result unless a
        // `;` discards it.
        if tokens.get(*pos) == Some(&ShaderToken::Ident("if".to_string())) {
            last = walk_if_statement(tokens, pos, ctx, pad, out)?;
            if tokens.get(*pos) == Some(&ShaderToken::Semi) {
                *pos += 1;
                last = None;
            }
            continue;
        }
        // A trailing expression: the result. If a `;` follows it is discarded.
        let v = walk_conditional(tokens, pos, ctx, pad, out)?;
        if tokens.get(*pos) == Some(&ShaderToken::Semi) {
            *pos += 1;
            last = None;
        } else {
            last = Some(v);
        }
    }
    Ok(last)
}

/// Consume a `{ … }` group starting at `pos`, returning the inner token slice
/// and advancing past the matching close brace. Mirrors the SPIR-V walker's
/// `take_braced`.
fn take_braced(tokens: &[ShaderToken], pos: &mut usize) -> Result<Vec<ShaderToken>, String> {
    if tokens.get(*pos) != Some(&ShaderToken::BraceOpen) {
        return Err("expected `{`".to_string());
    }
    *pos += 1;
    let start = *pos;
    let mut depth = 1i32;
    while *pos < tokens.len() && depth > 0 {
        match &tokens[*pos] {
            ShaderToken::BraceOpen => depth += 1,
            ShaderToken::BraceClose => depth -= 1,
            _ => {}
        }
        if depth > 0 {
            *pos += 1;
        }
    }
    let inner = tokens[start..*pos].to_vec();
    if *pos < tokens.len() {
        *pos += 1; // matching `}`
    }
    Ok(inner)
}

/// `if cond { … } [else { … }]` at statement level. Emits a WGSL `if`/`else`
/// block (WGSL has native statement control flow). Each branch runs the
/// statement walker over CLONED locals so a branch-local `let` stays scoped to
/// the branch; assignments to OUTER locals emit a WGSL assignment that persists
/// (the mutation is visible after the block, same as Rust).
///
/// When BOTH branches end in a trailing expression of the same type (the
/// `... } else { color }` tail shape), the `if` YIELDS that value: it is
/// lowered to a result `var` assigned in each arm, and the var name is returned
/// so the caller can make it the body result. Otherwise (no value, or only one
/// branch yields) the plain block is emitted and `None` is returned.
fn walk_if_statement(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<Option<(String, WType)>, String> {
    *pos += 1; // `if`
    let (cond, _) = walk_comparison(tokens, pos, ctx, pad, out)?;
    let then_tokens = take_braced(tokens, pos)?;
    let else_tokens = if tokens.get(*pos) == Some(&ShaderToken::Ident("else".to_string())) {
        *pos += 1;
        // `else if` chains parse as an `if` inside the else block; a braced
        // `{ … }` else is the common form. Both reduce to a braced group here.
        Some(take_braced(tokens, pos)?)
    } else {
        None
    };

    let inner_pad = format!("{pad}    ");

    // Walk each branch into its OWN buffer so we can learn whether it yields a
    // trailing value before deciding the block's shape.
    let mut then_ctx = Ctx {
        params: ctx.params,
        locals: ctx.locals.clone(),
        next_tmp: ctx.next_tmp,
    };
    let mut then_body = String::new();
    let mut tp = 0;
    let then_val = walk_statements(
        &then_tokens,
        &mut tp,
        &mut then_ctx,
        &inner_pad,
        &mut then_body,
    )?;

    let (else_body, else_val) = if let Some(et) = &else_tokens {
        let mut else_ctx = Ctx {
            params: ctx.params,
            locals: ctx.locals.clone(),
            next_tmp: ctx.next_tmp,
        };
        let mut eb = String::new();
        let mut ep = 0;
        let v = walk_statements(et, &mut ep, &mut else_ctx, &inner_pad, &mut eb)?;
        (Some(eb), v)
    } else {
        (None, None)
    };

    // Value form: both branches produced a trailing expression of one type.
    if let (Some((then_expr, then_ty)), Some((else_expr, else_ty))) = (&then_val, &else_val)
        && then_ty == else_ty
    {
        let else_body = else_body.expect("value form implies an else branch");
        let tmp = fresh_tmp(ctx);
        out.push_str(&format!("{pad}var {tmp}: {};\n", wgsl_type_name(*then_ty)));
        out.push_str(&format!("{pad}if ({cond}) {{\n"));
        out.push_str(&then_body);
        out.push_str(&format!("{inner_pad}{tmp} = {then_expr};\n"));
        out.push_str(&format!("{pad}}} else {{\n"));
        out.push_str(&else_body);
        out.push_str(&format!("{inner_pad}{tmp} = {else_expr};\n"));
        out.push_str(&format!("{pad}}}\n"));
        return Ok(Some((tmp, *then_ty)));
    }

    // Block form: emit the branches verbatim; any trailing expression a branch
    // produced (only one branch, or mismatched types) is discarded, matching
    // the SPIR-V walker's statement-`if`.
    out.push_str(&format!("{pad}if ({cond}) {{\n"));
    out.push_str(&then_body);
    out.push_str(&format!("{pad}}}"));
    if let Some(eb) = else_body {
        out.push_str(" else {\n");
        out.push_str(&eb);
        out.push_str(&format!("{pad}}}\n"));
    } else {
        out.push('\n');
    }
    Ok(None)
}

/// A conditional: either a value-form `if a { b } else { c }` (native WGSL,
/// emitted inline as `select`-free `if`-expression via a helper `var`) or a
/// plain comparison. WGSL has no ternary; a value-`if` whose branches are pure
/// expressions is lowered to a fresh `var` assigned in a statement-`if`, and
/// the var name is returned as the expression — so `let c = if … { … } else …`
/// stays legal WGSL. Branch bodies that carry their own `let`s / outer-local
/// assignments (the dija-rect shape) go through the statement walker too.
fn walk_conditional(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    if *pos < tokens.len() && tokens[*pos] == ShaderToken::Ident("if".to_string()) {
        *pos += 1;
        let (cond, _) = walk_comparison(tokens, pos, ctx, pad, out)?;
        let then_tokens = take_braced(tokens, pos)?;
        if tokens.get(*pos) != Some(&ShaderToken::Ident("else".to_string())) {
            return Err("if-expression without else is not supported".to_string());
        }
        *pos += 1; // `else`
        // The else arm is either a braced block or a chained `else if …`. A
        // chained `else if` is another value-`if` expression: recurse into it
        // as the single "statement" of a synthetic else block.
        let chained_else_if = tokens.get(*pos) == Some(&ShaderToken::Ident("if".to_string()));
        let else_tokens = if chained_else_if {
            // Take the rest of the `if … else …` chain as the else body.
            let start = *pos;
            // Consume a full conditional to find where the chain ends.
            let mut probe = *pos;
            skip_conditional(tokens, &mut probe)?;
            let slice = tokens[start..probe].to_vec();
            *pos = probe;
            slice
        } else {
            take_braced(tokens, pos)?
        };

        // Lower to a fresh var assigned inside a statement-`if`. The branch
        // bodies run the statement walker so branch-local `let`s and outer-
        // local assignments are honored; the branch's trailing expression is
        // the assigned value.
        let tmp = fresh_tmp(ctx);
        let inner_pad = format!("{pad}    ");

        // Then branch → value.
        let mut then_ctx = Ctx {
            params: ctx.params,
            locals: ctx.locals.clone(),
            next_tmp: ctx.next_tmp,
        };
        let mut then_body = String::new();
        let mut tp = 0;
        let (then_expr, then_ty) = walk_statements(
            &then_tokens,
            &mut tp,
            &mut then_ctx,
            &inner_pad,
            &mut then_body,
        )?
        .ok_or_else(|| "if-expression then-branch has no result value".to_string())?;

        // Else branch → value (a chained `else if` recurses through the same
        // conditional grammar).
        let mut else_ctx = Ctx {
            params: ctx.params,
            locals: ctx.locals.clone(),
            next_tmp: ctx.next_tmp,
        };
        let mut else_body = String::new();
        let mut ep = 0;
        let (else_expr, _else_ty) = walk_statements(
            &else_tokens,
            &mut ep,
            &mut else_ctx,
            &inner_pad,
            &mut else_body,
        )?
        .ok_or_else(|| "if-expression else-branch has no result value".to_string())?;

        // Declare the result var, then assign it in each arm.
        out.push_str(&format!("{pad}var {tmp}: {};\n", wgsl_type_name(then_ty)));
        out.push_str(&format!("{pad}if ({cond}) {{\n"));
        out.push_str(&then_body);
        out.push_str(&format!("{inner_pad}{tmp} = {then_expr};\n"));
        out.push_str(&format!("{pad}}} else {{\n"));
        out.push_str(&else_body);
        out.push_str(&format!("{inner_pad}{tmp} = {else_expr};\n"));
        out.push_str(&format!("{pad}}}\n"));
        return Ok((tmp, then_ty));
    }
    walk_comparison(tokens, pos, ctx, pad, out)
}

/// Advance `pos` past a full conditional (`if … else …`, else-if chains, or a
/// bare comparison) WITHOUT emitting — used to carve out the else body of an
/// `else if` chain so it can be re-walked as a nested value-`if`.
fn skip_conditional(tokens: &[ShaderToken], pos: &mut usize) -> Result<(), String> {
    if tokens.get(*pos) == Some(&ShaderToken::Ident("if".to_string())) {
        *pos += 1;
        // condition up to `{`
        while *pos < tokens.len() && tokens[*pos] != ShaderToken::BraceOpen {
            *pos += 1;
        }
        let _ = take_braced(tokens, pos)?;
        if tokens.get(*pos) == Some(&ShaderToken::Ident("else".to_string())) {
            *pos += 1;
            if tokens.get(*pos) == Some(&ShaderToken::Ident("if".to_string())) {
                return skip_conditional(tokens, pos);
            }
            let _ = take_braced(tokens, pos)?;
        }
        Ok(())
    } else {
        Err("expected `if` in else-if chain".to_string())
    }
}

/// A comparison: `a <cmp> b`, or just `a` when no comparison operator follows.
/// The result of a comparison is a WGSL `bool`; the shader grammar only uses it
/// as an `if` condition, so its type is reported as `F32` (it never feeds
/// arithmetic — same simplification the SPIR-V walker makes).
fn walk_comparison(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    let (left, ty) = walk_additive(tokens, pos, ctx, pad, out)?;
    if let Some(ShaderToken::Cmp(op)) = tokens.get(*pos) {
        let op = *op;
        *pos += 1;
        let (right, _) = walk_additive(tokens, pos, ctx, pad, out)?;
        return Ok((format!("{left} {} {right}", op.wgsl()), WType::F32));
    }
    Ok((left, ty))
}

/// Additive precedence: `a + b - c`. WGSL `+`/`-` map straight through.
fn walk_additive(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    let (mut left, mut ty) = walk_multiplicative(tokens, pos, ctx, pad, out)?;
    while let Some(tok) = tokens.get(*pos) {
        let op = match tok {
            ShaderToken::Op('+') => '+',
            ShaderToken::Op('-') => '-',
            _ => break,
        };
        *pos += 1;
        let (right, rty) = walk_multiplicative(tokens, pos, ctx, pad, out)?;
        // A vector±scalar or scalar±vector keeps the vector type; otherwise the
        // left type wins (both sides agree in practice).
        ty = join_arith_ty(ty, rty);
        left = format!("{left} {op} {right}");
    }
    Ok((left, ty))
}

/// Multiplicative precedence: `a * b / c`. A `mat * vec` in WGSL is written the
/// same as any other `*` (the language does the matrix-times-vector), so no
/// special opcode is needed — but the RESULT type must become the vector's so a
/// following swizzle validates. Division keeps the left type.
fn walk_multiplicative(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    let (mut left, mut ty) = walk_unary(tokens, pos, ctx, pad, out)?;
    while let Some(tok) = tokens.get(*pos) {
        let op = match tok {
            ShaderToken::Op('*') => '*',
            ShaderToken::Op('/') => '/',
            _ => break,
        };
        *pos += 1;
        let (right, rty) = walk_unary(tokens, pos, ctx, pad, out)?;
        if op == '*' {
            let is_left_mat = matches!(ty, WType::Mat4 | WType::Mat3);
            let is_right_vec = matches!(rty, WType::Vec4 | WType::Vec3);
            if is_left_mat && is_right_vec {
                ty = rty;
            } else {
                ty = join_arith_ty(ty, rty);
            }
        }
        left = format!("{left} {op} {right}");
    }
    Ok((left, ty))
}

/// A `vec op scalar` (or vice-versa) keeps the vector type; two equal types
/// keep that type; otherwise the left type is kept (the grammar never mixes
/// two different vector arities in one arithmetic op).
fn join_arith_ty(a: WType, b: WType) -> WType {
    match (a.vector_arity().is_some(), b.vector_arity().is_some()) {
        (true, _) => a,
        (false, true) => b,
        _ => a,
    }
}

/// Unary: leading `-` negation, and the uniform deref `*name`. A `&T` uniform
/// reads as `*viewport` / `(*viewport).x` in the source; WGSL binds the uniform
/// as a value, so the deref is a no-op — the star is dropped and the inner
/// expression parsed directly (parity with the SPIR-V walker's deref handling).
fn walk_unary(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    if tokens.get(*pos) == Some(&ShaderToken::Op('-')) {
        *pos += 1;
        let (val, ty) = walk_unary(tokens, pos, ctx, pad, out)?;
        return Ok((format!("-{val}"), ty));
    }
    if tokens.get(*pos) == Some(&ShaderToken::Op('*')) {
        *pos += 1;
        return walk_unary(tokens, pos, ctx, pad, out);
    }
    walk_atom(tokens, pos, ctx, pad, out)
}

/// Parse one atom, then apply any postfix swizzle (`.x`, `.zw`, …) — the
/// value-producing atoms (`Vec4::new(...)`, `sample(...)`, math calls,
/// parenthesized exprs, ctor results) all accept it, mirroring the SPIR-V
/// walker's `parse_atom` postfix loop.
fn walk_atom(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    let mut cur = walk_atom_inner(tokens, pos, ctx, pad, out)?;
    while tokens.get(*pos) == Some(&ShaderToken::Dot)
        && let Some(ShaderToken::Ident(field)) = tokens.get(*pos + 1)
    {
        if !is_swizzle(field) {
            // A `.` followed by a non-swizzle identifier that is itself
            // followed by `(` is a METHOD CALL (`uv.length()`), which the
            // shader grammar rejects — same as SPIR-V/MSL.
            if tokens.get(*pos + 2) == Some(&ShaderToken::Open) {
                return Err(format!(
                    "method call `.{field}(...)` is not supported in shader bodies"
                ));
            }
            return Err(format!("unknown field `.{field}`"));
        }
        let field = field.clone();
        *pos += 2;
        cur = apply_swizzle(&cur.0, cur.1, &field)?;
    }
    Ok(cur)
}

fn walk_atom_inner(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    let Some(tok) = tokens.get(*pos) else {
        return Err("unexpected end of expression".to_string());
    };
    match tok {
        ShaderToken::Float(v) => {
            *pos += 1;
            Ok((fmt_float(*v), WType::F32))
        }
        ShaderToken::Open => {
            *pos += 1;
            let (inner, ty) = walk_conditional(tokens, pos, ctx, pad, out)?;
            if tokens.get(*pos) == Some(&ShaderToken::Close) {
                *pos += 1;
            }
            Ok((format!("({inner})"), ty))
        }
        ShaderToken::Ident(name) => {
            let name = name.clone();
            *pos += 1;

            // Vec{2,3,4}::new(args) → vecN<f32>(args). Tolerates the wrap that
            // splits `Vec4 ::` / `new(` across a newline (the tokenizer has
            // already flattened the newline to whitespace).
            if (name == "Vec2" || name == "Vec3" || name == "Vec4")
                && tokens.get(*pos) == Some(&ShaderToken::ColonColon)
                && matches!(tokens.get(*pos + 1), Some(ShaderToken::Ident(n)) if n == "new")
            {
                return walk_vec_ctor(&name, tokens, pos, ctx, pad, out);
            }

            // Texture sampling: sample(slot, uv) → textureSample(tex_N, smp_N, uv)
            if name == "sample" && tokens.get(*pos) == Some(&ShaderToken::Open) {
                return walk_texture_sample(tokens, pos, ctx, pad, out);
            }

            // Screen-space derivatives — keep their WGSL names.
            if matches!(name.as_str(), "fwidth" | "dpdx" | "dpdy")
                && tokens.get(*pos) == Some(&ShaderToken::Open)
            {
                *pos += 1; // `(`
                let (arg, ty) = walk_conditional(tokens, pos, ctx, pad, out)?;
                consume_call_close(tokens, pos);
                return Ok((format!("{name}({arg})"), ty));
            }

            // Math intrinsics — call the WGSL builtin (same-name for most).
            if tokens.get(*pos) == Some(&ShaderToken::Open) {
                if let Some((wgsl_name, result_ty)) = wgsl_intrinsic(&name) {
                    return walk_intrinsic_call(wgsl_name, result_ty, tokens, pos, ctx, pad, out);
                }
                // A call to something the grammar does not know is a rejection
                // (parity with MSL/SPIR-V rejecting an unknown intrinsic).
                return Err(format!(
                    "unknown intrinsic or function `{name}` in shader body"
                ));
            }

            // Slice indexing: name[index] on a `&[T]` slice param.
            if tokens.get(*pos) == Some(&ShaderToken::BracketOpen) {
                return walk_slice_index(&name, tokens, pos, ctx, pad, out);
            }

            // Bare identifier: local, param, or boolean literal.
            walk_bare_ident(&name, ctx)
        }
        other => Err(format!("unexpected token: {other:?}")),
    }
}

/// A bare identifier resolves to a local/param (emitted verbatim) or a boolean
/// literal. An unknown name is an error — the grammar has no implicit globals.
fn walk_bare_ident(name: &str, ctx: &Ctx) -> Result<(String, WType), String> {
    if name == "true" {
        return Ok(("1.0".to_string(), WType::F32));
    }
    if name == "false" {
        return Ok(("0.0".to_string(), WType::F32));
    }
    match ctx.lookup(name) {
        Some(ty) => Ok((name.to_string(), ty)),
        None => Err(format!("unknown identifier `{name}`")),
    }
}

/// `VecN::new(a, b, …)` → `vecN<f32>(a, b, …)`. The argument list tolerates a
/// trailing comma (`1.0,)`) that rustfmt appends to wrapped calls.
fn walk_vec_ctor(
    name: &str,
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    *pos += 2; // `::` `new`
    let (count, out_ty, wgsl) = match name {
        "Vec2" => (2u32, WType::Vec2, "vec2<f32>"),
        "Vec3" => (3, WType::Vec3, "vec3<f32>"),
        "Vec4" => (4, WType::Vec4, "vec4<f32>"),
        _ => unreachable!(),
    };
    if tokens.get(*pos) == Some(&ShaderToken::Open) {
        *pos += 1;
    }
    let mut args = Vec::new();
    for i in 0..count {
        if i > 0 && tokens.get(*pos) == Some(&ShaderToken::Comma) {
            *pos += 1;
        }
        let (a, _) = walk_conditional(tokens, pos, ctx, pad, out)?;
        args.push(a);
    }
    consume_call_close(tokens, pos);
    Ok((format!("{wgsl}({})", args.join(", ")), out_ty))
}

/// `sample(slot, uv)` → `textureSample(tex_N, smp_N, uv)`. The slot must be a
/// literal number (matching the DSL's `sample(0, uv)` contract). The texture
/// and sampler bindings themselves are declared in `shader.rs` from a scan of
/// the body for `sample(N`.
fn walk_texture_sample(
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    *pos += 1; // `(`
    let slot = match tokens.get(*pos) {
        Some(ShaderToken::Float(f)) => {
            let s = *f as u32;
            *pos += 1;
            s
        }
        _ => return Err("sample() first arg must be a literal slot number".to_string()),
    };
    if tokens.get(*pos) == Some(&ShaderToken::Comma) {
        *pos += 1;
    }
    let (uv, _) = walk_conditional(tokens, pos, ctx, pad, out)?;
    consume_call_close(tokens, pos);
    Ok((
        format!("textureSample(tex_{slot}, smp_{slot}, {uv})"),
        WType::Vec4,
    ))
}

/// A math intrinsic call: `name(args)`. `dot`/`length`/`distance` return a
/// scalar; the rest return the type of their first argument (WGSL is
/// component-wise). Trailing commas are tolerated.
fn walk_intrinsic_call(
    wgsl_name: &str,
    result_ty: IntrinsicResult,
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    *pos += 1; // `(`
    let mut args = Vec::new();
    let mut first_ty = WType::F32;
    loop {
        if tokens.get(*pos) == Some(&ShaderToken::Close) {
            break;
        }
        if !args.is_empty() && tokens.get(*pos) == Some(&ShaderToken::Comma) {
            *pos += 1;
            if tokens.get(*pos) == Some(&ShaderToken::Close) {
                break; // trailing comma
            }
        }
        let (a, t) = walk_conditional(tokens, pos, ctx, pad, out)?;
        if args.is_empty() {
            first_ty = t;
        }
        args.push(a);
    }
    consume_call_close(tokens, pos);
    let ty = match result_ty {
        IntrinsicResult::Scalar => WType::F32,
        IntrinsicResult::FirstArg => first_ty,
    };
    Ok((format!("{wgsl_name}({})", args.join(", ")), ty))
}

/// `name[index]` on a `&[T]` slice param → `name[u32(index)]` (WGSL array
/// indices are integral; a computed `f32` index truncates, exactly as the MSL
/// `(uint)(index)` and the SPIR-V `OpConvertFToU`). Indexing a non-slice value
/// is a rejection.
fn walk_slice_index(
    name: &str,
    tokens: &[ShaderToken],
    pos: &mut usize,
    ctx: &mut Ctx,
    pad: &str,
    out: &mut String,
) -> Result<(String, WType), String> {
    let Some(slice) = ctx.slice(name) else {
        return Err(format!(
            "`{name}[..]` indexes a non-slice value; only `&[T]` slice params support indexing"
        ));
    };
    let elem_ty = slice.ty;
    *pos += 1; // `[`
    let (index, _) = walk_conditional(tokens, pos, ctx, pad, out)?;
    if tokens.get(*pos) == Some(&ShaderToken::BracketClose) {
        *pos += 1;
    } else {
        return Err(format!("expected `]` after index into `{name}`"));
    }
    Ok((format!("{name}[u32({index})]"), elem_ty))
}

/// Apply a component/swizzle (`.x`, `.zw`, `.rgba`, …) to a vector value,
/// validating each component against the source arity — an out-of-range
/// component (`.q`, or `.z` on a Vec2) is a rejection, matching SPIR-V/MSL.
/// WGSL swizzles are the same syntax as the source, so the value text just
/// gets `.field` appended; only validation and the result type are computed.
fn apply_swizzle(value: &str, ty: WType, field: &str) -> Result<(String, WType), String> {
    let Some(arity) = ty.vector_arity() else {
        return Err(format!(
            "cannot swizzle `.{field}` on a non-vector value ({ty:?})"
        ));
    };
    let indices: Vec<u32> = field
        .chars()
        .map(|c| component_index(c).ok_or_else(|| format!("unknown swizzle component `.{field}`")))
        .collect::<Result<_, _>>()?;
    if indices.is_empty() || indices.len() > 4 {
        return Err(format!("unsupported swizzle length: .{field}"));
    }
    if let Some(&bad) = indices.iter().find(|&&i| i >= arity) {
        let name = ['x', 'y', 'z', 'w'][bad as usize];
        return Err(format!(
            "swizzle component `{name}` out of range for {ty:?}"
        ));
    }
    let out_ty = match indices.len() {
        1 => WType::F32,
        2 => WType::Vec2,
        3 => WType::Vec3,
        _ => WType::Vec4,
    };
    Ok((format!("{value}.{field}"), out_ty))
}

/// Close a call's argument list: skip an optional trailing comma, then `)`.
/// rustfmt appends a trailing comma to wrapped calls and the token printer
/// preserves it, so `f(a, b,)` is an ordinary shape on the wire — every
/// call-shaped form routes its close through here (parity with SPIR-V).
fn consume_call_close(tokens: &[ShaderToken], pos: &mut usize) {
    if tokens.get(*pos) == Some(&ShaderToken::Comma) {
        *pos += 1;
    }
    if tokens.get(*pos) == Some(&ShaderToken::Close) {
        *pos += 1;
    }
}

/// Whether the result of a math intrinsic is a scalar or its first argument's
/// type — the only distinction the grammar needs for downstream swizzles.
#[derive(Clone, Copy)]
enum IntrinsicResult {
    Scalar,
    FirstArg,
}

/// Map a Rust intrinsic name to its WGSL builtin and result-type rule. Cross-
/// checked against the compute `MathFn` table (`helpers::math_fn_str`) and the
/// real WGSL spec: most names pass through unchanged; `inverse_sqrt`/`rsqrt`
/// map to `inverseSqrt`, `smooth_step` to `smoothstep`, and `step` stays
/// `step`. `dot`/`length`/`distance` return a scalar. An unknown name yields
/// `None`, which the caller turns into a rejection.
fn wgsl_intrinsic(name: &str) -> Option<(&'static str, IntrinsicResult)> {
    use IntrinsicResult::{FirstArg, Scalar};
    Some(match name {
        "sin" => ("sin", FirstArg),
        "cos" => ("cos", FirstArg),
        "tan" => ("tan", FirstArg),
        "asin" => ("asin", FirstArg),
        "acos" => ("acos", FirstArg),
        "atan" => ("atan", FirstArg),
        "atan2" => ("atan2", FirstArg),
        "sqrt" => ("sqrt", FirstArg),
        "inverseSqrt" | "inverse_sqrt" | "rsqrt" => ("inverseSqrt", FirstArg),
        "abs" => ("abs", FirstArg),
        "floor" => ("floor", FirstArg),
        "ceil" => ("ceil", FirstArg),
        "round" => ("round", FirstArg),
        "fract" => ("fract", FirstArg),
        "min" => ("min", FirstArg),
        "max" => ("max", FirstArg),
        "clamp" => ("clamp", FirstArg),
        "mix" => ("mix", FirstArg),
        "step" => ("step", FirstArg),
        "smoothstep" | "smooth_step" => ("smoothstep", FirstArg),
        "pow" => ("pow", FirstArg),
        "exp" => ("exp", FirstArg),
        "exp2" => ("exp2", FirstArg),
        "log" => ("log", FirstArg),
        "log2" => ("log2", FirstArg),
        "normalize" => ("normalize", FirstArg),
        "cross" => ("cross", FirstArg),
        "fma" => ("fma", FirstArg),
        "dot" => ("dot", Scalar),
        "length" => ("length", Scalar),
        "distance" => ("distance", Scalar),
        _ => return None,
    })
}

/// A unique temporary name for a value-`if` result. The counter is shared
/// across the whole body (and every cloned branch context), so `_wif0`,
/// `_wif1`, … never repeat — WGSL does not shadow across nested `var`
/// declarations, so a reused name in a sibling branch would be a redefinition
/// error. The leading underscore keeps them out of the user's identifier space.
fn fresh_tmp(ctx: &Ctx) -> String {
    let n = ctx.next_tmp.get();
    ctx.next_tmp.set(n + 1);
    format!("_wif{n}")
}

/// The WGSL type name for a value type (used to declare a value-`if` result var).
fn wgsl_type_name(ty: WType) -> &'static str {
    match ty {
        WType::F32 => "f32",
        WType::Vec2 => "vec2<f32>",
        WType::Vec3 => "vec3<f32>",
        WType::Vec4 => "vec4<f32>",
        WType::Mat4 => "mat4x4<f32>",
        WType::Mat3 => "mat3x3<f32>",
    }
}

/// Format an f32 as a WGSL float literal: a decimal point and the `f` suffix
/// are both required (`1.0f`, `0.5f`). `{:?}` gives the shortest round-tripping
/// decimal (always with a point for finite non-integers, and `Debug` for f32
/// prints `1.0` not `1`).
fn fmt_float(v: f32) -> String {
    format!("{v:?}f")
}

/// True when `field` is a pure component/swizzle run (`x`, `zw`, `rgba`, …).
fn is_swizzle(field: &str) -> bool {
    !field.is_empty() && field.len() <= 4 && field.chars().all(|c| component_index(c).is_some())
}

/// Component letter → vector index (`x`/`r` → 0 … `w`/`a` → 3).
fn component_index(c: char) -> Option<u32> {
    match c {
        'x' | 'r' => Some(0),
        'y' | 'g' => Some(1),
        'z' | 'b' => Some(2),
        'w' | 'a' => Some(3),
        _ => None,
    }
}
