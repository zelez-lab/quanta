//! `Rust kernel source → Lean Quanta.KRust.Kernel literal` emitter.
//!
//! Step **E.6** (route a) of the source-preservation track. Walks
//! a `syn::ItemFn` representing a `#[quanta::kernel]` function and
//! produces a Lean source file that defines the matching
//! `Quanta.KRust.Kernel` literal. Together with the proc macro's
//! existing `KernelOps` emission, this gives the *dual emission*
//! shape the route-(a) commitment depends on:
//!
//! ```
//!   #[quanta::kernel] fn k(...) { ... }
//!         │
//!         ├──▶  KernelOps wire bytes  (existing, runtime-bound)
//!         │
//!         └──▶  Quanta.KRust.Kernel   (new — this module)
//! ```
//!
//! The Lean-side `Quanta.KRust.Preservation` lemmas (T590-T5A7) +
//! the `Quanta.KRust.EndToEnd.t5b0_kernel_preservation` composition
//! state that the two emissions are consistent: applying
//! `KRust.translate` to the second view yields the same KernelOps
//! the macro produced from the first.
//!
//! Status: this commit lays down the scaffolding — a function that
//! takes a `syn::ItemFn` and returns a Lean source string for a
//! shallow subset of supported syntax (literals, paths, simple
//! binary ops, let bindings). The full coverage matching the proc
//! macro's `parse/expr.rs` + `parse/stmt.rs` lands incrementally
//! alongside the per-rule preservation lemmas; each new constructor
//! covered here is paired with discharging the corresponding
//! `sorry` in `Preservation.lean`.
//!
//! The CLI wiring (`quanta codegen krust <example>`) is a follow-up
//! commit; the unit test below demonstrates the path end-to-end on
//! a small kernel.

use std::fmt::Write;
use syn::{Expr, ExprLit, FnArg, ItemFn, Lit, Pat, Stmt};

/// Render a Lean string literal, escaping backslashes and quotes.
fn lean_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Render a literal expression as a `KRust.Lit` constructor invocation.
///
/// Currently covers integer literals (defaulting to `.u32` when the
/// suffix is missing — same heuristic the proc macro uses for
/// kernel-grid arithmetic). Floats and bools are placeholders
/// returning `none` until the full case coverage is wired alongside
/// the per-rule preservation lemmas.
fn lit_to_lean(lit: &Lit) -> Option<String> {
    match lit {
        Lit::Bool(b) => Some(format!("KRust.Lit.bool {}", b.value)),
        Lit::Int(i) => {
            let n: i64 = i.base10_parse().ok()?;
            // Default to u32 — the kernel surface treats unsuffixed
            // integers as workgroup-grid quantities, which are u32.
            Some(format!("KRust.Lit.int {} KRust.Scalar.u32", n))
        }
        Lit::Float(_f) => None, // TODO: wire alongside `Lit.float` case.
        _ => None,
    }
}

/// Walk a `syn::Expr` and render the matching `KRust.Expr`.
/// Returns `None` for unsupported constructors (the proc macro's
/// per-arm coverage is wider than this initial cut; expanding here
/// is paired with discharging the matching preservation lemma).
fn expr_to_lean(e: &Expr) -> Option<String> {
    match e {
        Expr::Lit(ExprLit { lit, .. }) => Some(format!("KRust.Expr.lit ({})", lit_to_lean(lit)?)),
        Expr::Path(p) => {
            let name = p.path.get_ident()?.to_string();
            Some(format!("KRust.Expr.path {}", lean_str(&name)))
        }
        Expr::Paren(par) => expr_to_lean(&par.expr),
        Expr::Binary(b) => {
            let op = match &b.op {
                syn::BinOp::Add(_) => "KRust.BinOp.add",
                syn::BinOp::Sub(_) => "KRust.BinOp.sub",
                syn::BinOp::Mul(_) => "KRust.BinOp.mul",
                syn::BinOp::Div(_) => "KRust.BinOp.div",
                syn::BinOp::Rem(_) => "KRust.BinOp.rem",
                syn::BinOp::Eq(_) => "KRust.BinOp.eq",
                syn::BinOp::Ne(_) => "KRust.BinOp.ne",
                syn::BinOp::Lt(_) => "KRust.BinOp.lt",
                syn::BinOp::Le(_) => "KRust.BinOp.le",
                syn::BinOp::Gt(_) => "KRust.BinOp.gt",
                syn::BinOp::Ge(_) => "KRust.BinOp.ge",
                _ => return None,
            };
            let lhs = expr_to_lean(&b.left)?;
            let rhs = expr_to_lean(&b.right)?;
            Some(format!("KRust.Expr.binary {op} ({lhs}) ({rhs})"))
        }
        _ => None,
    }
}

/// Walk a `syn::Stmt` and render the matching `KRust.Stmt`.
fn stmt_to_lean(s: &Stmt) -> Option<String> {
    match s {
        Stmt::Local(local) => {
            let name = match &local.pat {
                Pat::Ident(pi) => pi.ident.to_string(),
                Pat::Type(pt) => match &*pt.pat {
                    Pat::Ident(pi) => pi.ident.to_string(),
                    _ => return None,
                },
                _ => return None,
            };
            let init = local.init.as_ref()?;
            let rhs = expr_to_lean(&init.expr)?;
            Some(format!(
                "KRust.Stmt.letDecl {} none ({rhs})",
                lean_str(&name)
            ))
        }
        Stmt::Expr(e, _) => {
            let lean = expr_to_lean(e)?;
            Some(format!("KRust.Stmt.exprS ({lean})"))
        }
        _ => None,
    }
}

/// Convert a `syn::ItemFn` representing a `#[quanta::kernel]`
/// function into a Lean source file defining the matching
/// `Quanta.KRust.Kernel` literal.
///
/// Returns the full Lean source as a `String`, suitable for writing
/// to `specs/verify/lean/Quanta/KRust/Generated/<KernelName>.lean`
/// alongside the existing per-kernel `KernelOps` wire bytes.
pub fn emit_kernel_lean(func: &ItemFn) -> Result<String, &'static str> {
    let kernel_name = func.sig.ident.to_string();

    // Walk the params — flat-arg style only for the initial cut.
    let mut params: Vec<(String, &'static str)> = Vec::new();
    for (slot, arg) in func.sig.inputs.iter().enumerate() {
        let FnArg::Typed(pat_type) = arg else {
            return Err("self-receiver kernels not supported");
        };
        let name = match pat_type.pat.as_ref() {
            Pat::Ident(pi) => pi.ident.to_string(),
            _ => return Err("destructured patterns not supported"),
        };
        // Param kind discovery — a real implementation would
        // inspect `pat_type.ty` for `&[T]` / `&mut [T]` / scalar
        // and pick `fieldRead` / `fieldWrite` / `constant`. For
        // the initial cut, default to `constant`.
        let _ = slot;
        params.push((name, "KRust.ParamKind.constant"));
    }

    // Walk the body.
    let stmts: Result<Vec<String>, &'static str> = func
        .block
        .stmts
        .iter()
        .map(|s| stmt_to_lean(s).ok_or("unsupported statement in initial-cut emitter"))
        .collect();
    let stmts = stmts?;

    // Render the Lean file.
    let mut out = String::new();
    writeln!(
        &mut out,
        "/- GENERATED — DO NOT EDIT. Emitted by `quanta-codegen::emit_krust`."
    )
    .unwrap();
    writeln!(&mut out, "Source: `#[quanta::kernel] fn {kernel_name}`.").unwrap();
    writeln!(
        &mut out,
        "Route-(a) dual view: this Lean literal is the source-side"
    )
    .unwrap();
    writeln!(
        &mut out,
        "shadow of the proc macro's KernelOps wire emission. The"
    )
    .unwrap();
    writeln!(
        &mut out,
        "preservation theorems in `Quanta.KRust.Preservation` link"
    )
    .unwrap();
    writeln!(&mut out, "the two by structural induction.").unwrap();
    writeln!(&mut out, "-/").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "import Quanta.KRust.Syntax").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "namespace Quanta.KRust.Generated").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "def {kernel_name} : KRust.Kernel :=").unwrap();
    writeln!(&mut out, "  {{ name := {},", lean_str(&kernel_name)).unwrap();
    writeln!(&mut out, "    params := [").unwrap();
    for (i, (pname, kind)) in params.iter().enumerate() {
        let comma = if i + 1 < params.len() { "," } else { "" };
        writeln!(
            &mut out,
            "      {{ name := {}, kind := {kind}, slot := {i}, scalarType := KRust.Scalar.u32 }}{comma}",
            lean_str(pname),
        )
        .unwrap();
    }
    writeln!(&mut out, "    ],").unwrap();
    writeln!(&mut out, "    workgroupSize := (1, 1, 1),").unwrap();
    writeln!(&mut out, "    body := [").unwrap();
    for (i, stmt) in stmts.iter().enumerate() {
        let comma = if i + 1 < stmts.len() { "," } else { "" };
        writeln!(&mut out, "      {stmt}{comma}").unwrap();
    }
    writeln!(&mut out, "    ] }}").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "end Quanta.KRust.Generated").unwrap();

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_minimal_kernel() {
        let src: syn::ItemFn = syn::parse_str(
            r#"
            fn add_one_demo(n: u32) {
                let x = n;
                let y = x + 1u32;
            }
            "#,
        )
        .unwrap();

        let out = emit_kernel_lean(&src).unwrap();
        // Sanity: the kernel name and a few structural strings appear.
        assert!(out.contains("def add_one_demo : KRust.Kernel"));
        assert!(out.contains("name := \"n\""));
        assert!(out.contains("KRust.Stmt.letDecl \"x\""));
        assert!(out.contains("KRust.Stmt.letDecl \"y\""));
        assert!(out.contains("KRust.BinOp.add"));
    }
}
