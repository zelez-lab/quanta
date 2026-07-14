//! The `crate = <path>` override for compute-face macros.
//!
//! Every path the compute DSL emits into user code splits into two
//! families:
//!
//! - **runtime / data-model types** — `KernelBinary`, `GpuType`,
//!   `ScalarType`, `Gpu`, `Wave`, `Pulse`, `QuantaError`,
//!   `__device_host_stubs`. These live in `quanta-core` and are
//!   re-exported by the `quanta` facade, so both `::quanta::Wave` and
//!   `::quanta_core::Wave` resolve the same type. The override picks
//!   which crate root to name them through.
//!
//! - **the compute-DSL proc-macros themselves** — `__kernel_inner`
//!   (emitted by `kernel`'s two-pass delegation) and
//!   `device(register_only)` (emitted inside a device fn's `_src!`
//!   macro). These live in *this* crate, `quanta_compute_dsl`, and are
//!   re-exported by the facade. When the default `::quanta` root is in
//!   force we keep naming them through the facade (so a facade-only
//!   consumer needs nothing else on their dependency line); when the
//!   root is overridden we name them through `::quanta_compute_dsl`
//!   directly (the override exists precisely so the caller depends on
//!   the split crates, not the facade).
//!
//! `#[quanta::kernel(crate = quanta_core)]` and the derive container
//! attribute `#[quanta(crate = quanta_core)]` both feed
//! [`CratePath::from_...`]; the single [`CratePath`] value is then the
//! one source of truth every emitter routes its paths through.

use proc_macro::TokenStream as RawTokenStream;
use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::{Attribute, Meta, Path, Token, punctuated::Punctuated};

/// The resolved crate roots the emitters name types and macros through.
#[derive(Clone)]
pub(crate) struct CratePath {
    /// Root for runtime / data-model types (`quanta-core` surface).
    /// Default `::quanta`; overridden to e.g. `quanta_core`.
    types: Path,
    /// True when the caller supplied an explicit `crate = ...`. Drives
    /// the proc-macro self-reference root (facade vs
    /// `quanta_compute_dsl`).
    overridden: bool,
}

impl Default for CratePath {
    fn default() -> Self {
        Self {
            types: syn::parse_quote!(::quanta),
            overridden: false,
        }
    }
}

impl CratePath {
    /// Build from an explicit user-supplied crate path.
    fn from_path(path: Path) -> Self {
        Self {
            types: path,
            overridden: true,
        }
    }

    /// Token stream naming a runtime / data-model type root, e.g.
    /// `::quanta` or `quanta_core`.
    pub(crate) fn types(&self) -> TokenStream {
        let p = &self.types;
        quote! { #p }
    }

    /// Token stream naming the compute-DSL proc-macro root. Facade
    /// (`::quanta`) by default; `::quanta_compute_dsl` once the caller
    /// overrode the crate root (they no longer depend on the facade).
    pub(crate) fn macros(&self) -> TokenStream {
        if self.overridden {
            quote! { ::quanta_compute_dsl }
        } else {
            quote! { ::quanta }
        }
    }
}

/// Pull a `crate = <path>` entry out of a parsed meta list, returning
/// the override (if present). Used by the attribute macros, which parse
/// their args as a `Punctuated<Meta, Comma>`.
pub(crate) fn crate_from_metas(metas: &Punctuated<Meta, Token![,]>) -> Option<CratePath> {
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

/// Parse an attribute macro's raw arg token stream (the tokens inside
/// `#[quanta::gpu_type(...)]`) into a `CratePath`, returning the default
/// `::quanta` root when there is no `crate = ...` entry (or no args at
/// all). Unknown args are ignored here — `gpu_type` takes no other
/// arguments today.
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
/// derive input's attribute list (serde's `#[serde(crate = "...")]`
/// pattern, but with a path value rather than a string). Returns the
/// default `::quanta` root when the attribute is absent.
pub(crate) fn crate_from_container_attrs(attrs: &[Attribute]) -> CratePath {
    for attr in attrs {
        if !attr.path().is_ident("quanta") {
            continue;
        }
        // `#[quanta(crate = some::path)]` — parse the parenthesised
        // body as a comma list of metas and look for `crate = ...`.
        if let Ok(list) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            && let Some(cp) = crate_from_metas(&list)
        {
            return cp;
        }
    }
    CratePath::default()
}
