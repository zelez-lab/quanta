//! Implementation body for `#[quanta::device]`.

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::ItemFn;

/// Core implementation of `#[quanta::device]`.
pub(crate) fn expand_device(func: ItemFn) -> TokenStream {
    let source = func.to_token_stream().to_string();
    let fn_name = &func.sig.ident;
    let const_name = syn::Ident::new(
        &format!("__QUANTA_DEVICE_{}", fn_name.to_string().to_uppercase()),
        fn_name.span(),
    );

    let expanded = quote! {
        pub const #const_name: &str = #source;
    };
    expanded.into()
}
