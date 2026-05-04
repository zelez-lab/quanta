//! WASM-twin emitter: produces a `#[cfg(target_arch = "wasm32")]`
//! `extern "C"` flattened copy of a struct-ref kernel.
//!
//! The twin is what rustc lowers to `wasm32-unknown-unknown` and the
//! WASM → KernelOps lowering pass consumes (roadmap step 058 phase 1.2).
//!
//! Today nothing reads the twin yet — the legacy syntax-tree parser
//! still drives translation. The twin is dormant data until the
//! lowering pass lands in step 2.2. We emit it now so:
//! 1. Every kernel is forced to be wasm32-compilable from day one
//!    (the twin breaks the build if it isn't), keeping us honest as
//!    the kernel surface grows.
//! 2. The lowering pass in step 2.2 has working input on day one.
//!
//! The flattening rules:
//! - Struct-ref param `d: &MyData` → one parameter per field. Buffer
//!   fields become `*const T` / `*mut T`; scalar fields become `T`.
//! - `d.field[idx]` → `*field.add(idx as usize)` (read) or
//!   `*field.add(idx as usize) = ...` (write).
//! - `d.scalar` → `scalar` (the flat local parameter).
//! - Intrinsic calls (`quark_id()`, `local_id()`, ...) need
//!   `use quanta::intrinsics::*` injected at the top of the body.
//!
//! Anything we can't flatten today emits a `compile_error!` so the
//! kernel surface and the wasm-twin emitter stay in sync.

#![allow(clippy::collapsible_if, clippy::needless_return)]

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::visit_mut::{self, VisitMut};
use syn::{Expr, ExprField, ExprIndex, ExprPath, ItemFn, Member};

use crate::auto_dispatch::{StructFieldAccess, StructParamInfo};

/// Emit the `#[cfg(target_arch = "wasm32")] extern "C" fn` twin.
///
/// `func` is the original user-written kernel; `info` carries the
/// struct-ref param name, the discovered field accesses, and per-field
/// scalar types.
pub(crate) fn emit_wasm_twin(func: &ItemFn, info: &StructParamInfo) -> TokenStream {
    let kernel_name = &func.sig.ident;

    // Build the parameter list — buffers first (in slot order), then
    // scalars. Slots are already assigned in `info.fields`.
    let mut buffer_fields: Vec<&StructFieldAccess> =
        info.fields.iter().filter(|f| f.is_indexed).collect();
    let mut scalar_fields: Vec<&StructFieldAccess> =
        info.fields.iter().filter(|f| !f.is_indexed).collect();
    buffer_fields.sort_by_key(|f| f.slot);
    scalar_fields.sort_by_key(|f| f.slot);

    let mut params = Vec::new();
    for f in &buffer_fields {
        let ident = format_ident!("{}", f.name);
        let ty = scalar_to_rust_ty(&f.scalar_type_name);
        if f.is_written {
            params.push(quote! { #ident: *mut #ty });
        } else {
            params.push(quote! { #ident: *const #ty });
        }
    }
    for f in &scalar_fields {
        let ident = format_ident!("{}", f.name);
        let ty = scalar_to_rust_ty(&f.scalar_type_name);
        params.push(quote! { #ident: #ty });
    }

    // Rewrite the body: replace every `d.field[idx]` and `d.scalar`
    // with the flat-parameter form.
    let mut rewriter = StructRefRewriter {
        param_name: info.param_name.clone(),
        buffer_field_names: buffer_fields.iter().map(|f| f.name.clone()).collect(),
        scalar_field_names: scalar_fields.iter().map(|f| f.name.clone()).collect(),
    };

    let mut body_block = func.block.clone();
    rewriter.visit_block_mut(&mut body_block);

    quote! {
        // The wasm-twin: rustc lowers this to wasm32 when the user's
        // crate is built for `wasm32-unknown-unknown`. Other targets
        // skip it entirely. Once step 2.2 lands, the lowering pass
        // reads this function out of the resulting WASM and produces
        // a `KernelDef` from it. Until then it's dormant data — but
        // it must compile, which keeps the kernel surface honest.
        #[cfg(target_arch = "wasm32")]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #kernel_name(#(#params),*) {
            #[allow(unused_imports)]
            use ::quanta::intrinsics::*;
            unsafe { #body_block }
        }
    }
}

/// Rewrite `d.field[idx]` → `*field.add(idx as usize)` and
/// `d.scalar` → `scalar` in place. Anything else is left unchanged
/// — rustc's typechecker enforces that the result still compiles.
struct StructRefRewriter {
    param_name: String,
    buffer_field_names: Vec<String>,
    scalar_field_names: Vec<String>,
}

impl StructRefRewriter {
    fn is_buffer_field(&self, name: &str) -> bool {
        self.buffer_field_names.iter().any(|n| n == name)
    }
    fn is_scalar_field(&self, name: &str) -> bool {
        self.scalar_field_names.iter().any(|n| n == name)
    }
    fn extract_field(&self, expr: &Expr) -> Option<String> {
        let Expr::Field(ExprField {
            base,
            member: Member::Named(ident),
            ..
        }) = expr
        else {
            return None;
        };
        let Expr::Path(ExprPath { path, .. }) = base.as_ref() else {
            return None;
        };
        let seg = path.segments.last()?;
        if seg.ident != self.param_name.as_str() {
            return None;
        }
        Some(ident.to_string())
    }
}

impl VisitMut for StructRefRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // Visit children first so deeper rewrites land before we
        // inspect the current node.
        visit_mut::visit_expr_mut(self, expr);

        // Replace `d.field[idx]` (buffer index) with `*field.add(idx as usize)`.
        if let Expr::Index(ExprIndex {
            expr: base, index, ..
        }) = expr
        {
            if let Some(field_name) = self.extract_field(base) {
                if self.is_buffer_field(&field_name) {
                    let ident = format_ident!("{}", field_name);
                    let idx = index.clone();
                    *expr = syn::parse_quote! {
                        *(#ident.add((#idx) as usize))
                    };
                    return;
                }
            }
        }

        // Replace `d.scalar` with `scalar`.
        if let Expr::Field(ExprField {
            member: Member::Named(_),
            ..
        }) = expr
        {
            if let Some(field_name) = self.extract_field(expr) {
                if self.is_scalar_field(&field_name) {
                    let ident = format_ident!("{}", field_name);
                    *expr = Expr::Path(syn::ExprPath {
                        attrs: Vec::new(),
                        qself: None,
                        path: syn::Path::from(ident),
                    });
                    return;
                }
            }
        }
    }
}

/// Map a Quanta scalar type name (the `KernelDef.params[i].scalar_type`
/// stringified) to a Rust type token.
fn scalar_to_rust_ty(name: &str) -> TokenStream {
    match name {
        "u8" => quote! { u8 },
        "u16" => quote! { u16 },
        "u32" => quote! { u32 },
        "u64" => quote! { u64 },
        "i8" => quote! { i8 },
        "i16" => quote! { i16 },
        "i32" => quote! { i32 },
        "i64" => quote! { i64 },
        "f16" => quote! { ::core::primitive::f32 }, // lower f16 → f32 on wasm32
        "f32" => quote! { f32 },
        "f64" => quote! { f64 },
        "bool" => quote! { bool },
        other => {
            // Unknown — emit a compile_error so we notice the gap.
            let msg =
                format!("wasm-twin: unsupported scalar type `{other}` — extend scalar_to_rust_ty");
            quote_spanned(msg)
        }
    }
}

fn quote_spanned(msg: String) -> TokenStream {
    let span = Span::call_site();
    let lit = syn::LitStr::new(&msg, span);
    quote! { compile_error!(#lit) }
}
