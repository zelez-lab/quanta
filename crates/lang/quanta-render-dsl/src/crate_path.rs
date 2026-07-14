//! The `crate = <path>` override for render-face macros.
//!
//! Every path the render DSL emits — `ShaderBinary`, `ShaderStage`,
//! `VertexAttribute`, `VertexLayout`, `StepMode`, `AttributeFormat` —
//! names a render data-model type that lives in `quanta-core` (behind
//! `render`) and is re-exported by the `quanta` facade, so both
//! `::quanta::ShaderBinary` and `::quanta_core::ShaderBinary` resolve
//! the same type. The override picks which crate root to name them
//! through. Default `::quanta`; a render consumer that depends on the
//! split crates rather than the facade passes `crate = quanta_core`.
//!
//! Symmetric with `quanta-compute-dsl::crate_path`, minus the
//! proc-macro self-reference machinery — the render macros emit no
//! self-referencing macro invocations, so a single "types root" is all
//! that is threaded.

use proc_macro::TokenStream as RawTokenStream;
use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::{Attribute, Meta, Path, Token, punctuated::Punctuated};

/// The resolved crate root render types are named through.
#[derive(Clone)]
pub(crate) struct CratePath {
    types: Path,
}

impl Default for CratePath {
    fn default() -> Self {
        Self {
            types: syn::parse_quote!(::quanta),
        }
    }
}

impl CratePath {
    fn from_path(path: Path) -> Self {
        Self { types: path }
    }

    /// Token stream naming a render data-model type root.
    pub(crate) fn types(&self) -> TokenStream {
        let p = &self.types;
        quote! { #p }
    }
}

fn crate_from_metas(metas: &Punctuated<Meta, Token![,]>) -> Option<CratePath> {
    for meta in metas {
        if let Meta::NameValue(nv) = meta
            && nv.path.is_ident("crate")
            && let syn::Expr::Path(ep) = &nv.value
        {
            return Some(CratePath::from_path(ep.path.clone()));
        }
    }
    None
}

/// Parse a shader-stage attribute macro's raw arg tokens (the tokens
/// inside `#[quanta::vertex(...)]`) into a `CratePath`, defaulting to
/// `::quanta` when there is no `crate = ...` entry.
pub(crate) fn from_attr_args(attr: RawTokenStream) -> CratePath {
    if attr.is_empty() {
        return CratePath::default();
    }
    let parser = Punctuated::<Meta, Token![,]>::parse_terminated;
    match parser.parse(attr) {
        Ok(metas) => crate_from_metas(&metas).unwrap_or_default(),
        Err(_) => CratePath::default(),
    }
}

/// Read the `#[quanta(crate = <path>)]` container attribute off a
/// derive input's attribute list (serde pattern). Returns the default
/// `::quanta` root when absent.
pub(crate) fn crate_from_container_attrs(attrs: &[Attribute]) -> CratePath {
    for attr in attrs {
        if !attr.path().is_ident("quanta") {
            continue;
        }
        if let Ok(list) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            && let Some(cp) = crate_from_metas(&list)
        {
            return cp;
        }
    }
    CratePath::default()
}
