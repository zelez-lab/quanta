//! `#[derive(ParamTree)]` — the typed-tree plumbing, generated.
//!
//! The hand-written `ParamTree` impls in `quanta-nn` proved the trait
//! shapes; this derive removes the boilerplate for user-defined trees:
//!
//! ```ignore
//! #[derive(ParamTree)]
//! pub struct BlockParams<T: DiffScalar> {
//!     pub attn: LinearParams<T>,
//!     pub norm: NormParams<T>,
//!     pub gate: Option<Array<T>>,
//! }
//! ```
//!
//! generates the `BlockParamsVars<T>` twin (same tree shape, `Vars`
//! leaves) and the `ParamTree` impl — `bind`, order-stable
//! `flatten`/`unflatten`, and `grads` — by delegating to each field in
//! declaration order. Every field type must itself implement
//! `ParamTree<T>` (`Array<T>`, `Option<…>`, the shipped param structs,
//! and tuples all do).
//!
//! Requirements: a struct with named fields and exactly one type
//! parameter (the scalar). Inside `quanta-nn` itself, point the
//! generated paths at the crate root with `#[param_tree(crate = crate)]`
//! (the same convention as `#[quanta_compute_dsl::kernel(crate = …)]`).

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(ParamTree, attributes(param_tree))]
pub fn derive_param_tree(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let mut root: syn::Path = syn::parse_quote!(quanta_nn);
    for attr in &input.attrs {
        if attr.path().is_ident("param_tree") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("crate") {
                    root = meta.value()?.parse()?;
                    Ok(())
                } else {
                    Err(meta.error("supported: #[param_tree(crate = path)]"))
                }
            })?;
        }
    }

    let name = &input.ident;
    let vis = &input.vis;
    let vars_name = format_ident!("{}Vars", name);

    let type_params: Vec<_> = input.generics.type_params().collect();
    if type_params.len() != 1 {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "#[derive(ParamTree)] needs exactly one type parameter (the scalar)",
        ));
    }
    let tp = type_params[0].ident.clone();

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => f.named.clone(),
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "#[derive(ParamTree)] supports named-field structs",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "#[derive(ParamTree)] supports structs only",
            ));
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let field_tys: Vec<_> = fields.iter().map(|f| f.ty.clone()).collect();
    let fnames: Vec<_> = fields.iter().map(|f| f.ident.clone().unwrap()).collect();
    let fvis: Vec<_> = fields.iter().map(|f| f.vis.clone()).collect();

    let mut predicates: Vec<syn::WherePredicate> = Vec::new();
    if let Some(w) = where_clause {
        predicates.extend(w.predicates.iter().cloned());
    }
    for ty in &field_tys {
        predicates.push(syn::parse_quote!(#ty: #root::layer::ParamTree<#tp>));
    }
    let where_tok = if predicates.is_empty() {
        quote! {}
    } else {
        quote! { where #(#predicates,)* }
    };

    Ok(quote! {
        #vis struct #vars_name #impl_generics #where_tok {
            #( #fvis #fnames: <#field_tys as #root::layer::ParamTree<#tp>>::Vars, )*
        }

        impl #impl_generics #root::layer::ParamTree<#tp> for #name #ty_generics #where_tok {
            type Vars = #vars_name #ty_generics;

            fn bind(&self, tape: &#root::Tape<#tp>) -> Self::Vars {
                #vars_name {
                    #( #fnames: #root::layer::ParamTree::<#tp>::bind(&self.#fnames, tape), )*
                }
            }

            fn flatten(&self) -> ::std::vec::Vec<#root::Array<#tp>> {
                let mut v = ::std::vec::Vec::new();
                #( v.extend(#root::layer::ParamTree::<#tp>::flatten(&self.#fnames)); )*
                v
            }

            fn unflatten(
                &self,
                leaves: &mut ::std::vec::IntoIter<#root::Array<#tp>>,
            ) -> ::core::result::Result<Self, #root::AutogradError> {
                ::core::result::Result::Ok(#name {
                    #( #fnames: #root::layer::ParamTree::<#tp>::unflatten(&self.#fnames, leaves)?, )*
                })
            }

            fn grads(
                vars: &Self::Vars,
                loss: &#root::Var<#tp>,
            ) -> ::core::result::Result<Self, #root::AutogradError> {
                ::core::result::Result::Ok(#name {
                    #( #fnames: <#field_tys as #root::layer::ParamTree<#tp>>::grads(&vars.#fnames, loss)?, )*
                })
            }
        }
    })
}
