//! Implementation of `#[derive(Varyings)]` — the shared vertex↔fragment
//! interface struct of the render DSL.
//!
//! A Varyings struct is the single explicit interface between a vertex and a
//! fragment shader: exactly one `#[position]` field of type `Vec4` (routed
//! to gl_Position / `[[position]]`; reading it in a fragment yields the
//! interpolated window position), plus zero or more varying fields —
//! `f32` / `u32` / `Vec2` / `Vec3` / `Vec4` — assigned Location 0, 1, … in
//! FIELD-DECLARATION order (`u32` fields are flat-interpolated on both
//! interface ends).
//!
//! # What the derive generates
//!
//! 1. **Introspection consts** on the struct (`POSITION_FIELD`,
//!    `VARYING_FIELDS` — name/type-name pairs in Location order), so host
//!    code and tests can see the interface without re-parsing anything.
//!
//! 2. **The interface trampoline** — `macro_rules! __quanta_varyings_<Name>`
//!    plus a same-visibility `use` re-export beside the struct. This is how
//!    the field metadata CROSSES proc-macro invocations: a proc macro can
//!    only see the item it is attached to, so `#[quanta::vertex]` /
//!    `#[quanta::fragment]` (which see `fn vs(...) -> Surface`, not the
//!    struct) expand to an invocation of this trampoline, which pastes the
//!    field list in front of the function tokens and forwards everything to
//!    the hidden second-stage proc-macro (`__vertex_varyings` /
//!    `__fragment_varyings`) that runs the real shader compile. Definition
//!    order therefore matters the way it does in any macro_rules scope:
//!    declare the Varyings struct BEFORE the shaders that use it (or import
//!    the generated `__quanta_varyings_<Name>` alongside the struct when
//!    they live in another module).

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Fields, ItemStruct};

/// The shader-interface field types a Varyings struct may declare, by type
/// NAME (the derive matches the last path segment, like every DSL surface —
/// the canonical types are re-exported at the crate root: `quanta::Vec2`…).
const FIELD_TYPES: &[&str] = &["f32", "u32", "Vec2", "Vec3", "Vec4"];

/// One parsed field: its name, its type name, and whether it carries the
/// `#[position]` marker.
struct VField {
    name: syn::Ident,
    ty_name: String,
    is_position: bool,
}

/// Generate the `Varyings` derive implementation for a struct.
pub(crate) fn expand_varyings_derive(input: &ItemStruct) -> Result<TokenStream, syn::Error> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "#[derive(Varyings)] does not support generic structs",
        ));
    }
    let fields = match &input.fields {
        Fields::Named(named) => &named.named,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "#[derive(Varyings)] only supports structs with named fields",
            ));
        }
    };

    let mut parsed: Vec<VField> = Vec::new();
    for field in fields {
        let name = field
            .ident
            .clone()
            .ok_or_else(|| syn::Error::new_spanned(field, "unnamed field"))?;
        let is_position = field.attrs.iter().any(|a| a.path().is_ident("position"));
        let ty_name = match &field.ty {
            syn::Type::Path(p) => p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default(),
            _ => String::new(),
        };
        if is_position {
            if ty_name != "Vec4" {
                return Err(syn::Error::new_spanned(
                    &field.ty,
                    "the #[position] field must be a Vec4 (it becomes gl_Position)",
                ));
            }
        } else if !FIELD_TYPES.contains(&ty_name.as_str()) {
            return Err(syn::Error::new_spanned(
                &field.ty,
                format!(
                    "field `{name}`: unsupported varying type `{ty_name}`. \
                     Supported: f32, u32, Vec2, Vec3, Vec4"
                ),
            ));
        }
        parsed.push(VField {
            name,
            ty_name,
            is_position,
        });
    }

    let position_count = parsed.iter().filter(|f| f.is_position).count();
    if position_count != 1 {
        return Err(syn::Error::new_spanned(
            &input.ident,
            format!(
                "#[derive(Varyings)] requires exactly one #[position] field \
                 (the clip-space position), found {position_count}"
            ),
        ));
    }

    let struct_name = &input.ident;
    let vis = &input.vis;
    let position = parsed
        .iter()
        .find(|f| f.is_position)
        .expect("checked above");
    let position_name = position.name.to_string();
    let varyings: Vec<&VField> = parsed.iter().filter(|f| !f.is_position).collect();

    // Introspection consts: the position field's name and the varying
    // (name, type-name) pairs in Location order.
    let varying_entries: Vec<TokenStream> = varyings
        .iter()
        .map(|f| {
            let n = f.name.to_string();
            let t = &f.ty_name;
            quote! { (#n, #t) }
        })
        .collect();
    let varying_count = varying_entries.len();

    // The trampoline: pastes the interface metadata (every field in
    // declaration order, the position one marked `#[position]`) in front of
    // whatever the callback macro is handed. Field-declaration order is
    // preserved verbatim so Location assignment is deterministic.
    let mac_name = format_ident!("__quanta_varyings_{}", struct_name);
    let meta_fields: Vec<TokenStream> = parsed
        .iter()
        .map(|f| {
            let name = &f.name;
            let ty = format_ident!("{}", f.ty_name);
            if f.is_position {
                quote! { #[position] #name : #ty }
            } else {
                quote! { #name : #ty }
            }
        })
        .collect();

    let all_field_names: Vec<&syn::Ident> = parsed.iter().map(|f| &f.name).collect();

    Ok(quote! {
        impl #struct_name {
            /// The `#[position]` field's name — generated by
            /// `#[derive(Varyings)]`.
            pub const POSITION_FIELD: &'static str = #position_name;

            /// The varying fields as `(name, type-name)` pairs, in
            /// field-declaration order: entry `i` is Location `i` on every
            /// backend. Generated by `#[derive(Varyings)]`.
            pub const VARYING_FIELDS: [(&'static str, &'static str); #varying_count] = [
                #(#varying_entries),*
            ];

            /// Marks every field as read so a Varyings struct used only as a
            /// shader interface (the macros erase the shader fns, so host
            /// code never touches the fields) doesn't trip `dead_code`.
            #[doc(hidden)]
            #[allow(dead_code)]
            fn __quanta_varyings_fields_are_the_interface(&self) {
                #(let _ = &self.#all_field_names;)*
            }
        }

        #[doc(hidden)]
        macro_rules! #mac_name {
            (($($cb:tt)*) $($rest:tt)*) => {
                $($cb)* ! {
                    @varyings #struct_name { #(#meta_fields),* }
                    $($rest)*
                }
            };
        }
        #[doc(hidden)]
        #[allow(unused_imports)]
        #vis use #mac_name;
    })
}

#[cfg(test)]
mod tests {
    use super::expand_varyings_derive;

    fn expand(src: &str) -> Result<String, String> {
        let item: syn::ItemStruct = syn::parse_str(src).expect("test struct parses");
        expand_varyings_derive(&item)
            .map(|ts| ts.to_string())
            .map_err(|e| e.to_string())
    }

    #[test]
    fn generates_consts_and_trampoline() {
        let out = expand("struct Surface { #[position] clip: Vec4, uv: Vec2, kind: u32 }").unwrap();
        assert!(out.contains("POSITION_FIELD"), "out: {out}");
        assert!(out.contains("\"clip\""), "out: {out}");
        assert!(out.contains("(\"uv\" , \"Vec2\")"), "out: {out}");
        assert!(out.contains("(\"kind\" , \"u32\")"), "out: {out}");
        assert!(out.contains("__quanta_varyings_Surface"), "out: {out}");
        // The trampoline metadata preserves declaration order and the
        // position marker.
        assert!(
            out.contains(
                "@ varyings Surface { # [position] clip : Vec4 , uv : Vec2 , kind : u32 }"
            ),
            "out: {out}"
        );
    }

    #[test]
    fn requires_exactly_one_position() {
        let err = expand("struct S { uv: Vec2 }").unwrap_err();
        assert!(err.contains("exactly one #[position]"), "err: {err}");
        let err = expand("struct S { #[position] a: Vec4, #[position] b: Vec4 }").unwrap_err();
        assert!(err.contains("exactly one #[position]"), "err: {err}");
    }

    #[test]
    fn position_must_be_vec4() {
        let err = expand("struct S { #[position] clip: Vec3, uv: Vec2 }").unwrap_err();
        assert!(err.contains("must be a Vec4"), "err: {err}");
    }

    #[test]
    fn rejects_unsupported_field_types() {
        let err = expand("struct S { #[position] clip: Vec4, m: Mat4 }").unwrap_err();
        assert!(
            err.contains("unsupported varying type `Mat4`"),
            "err: {err}"
        );
    }

    #[test]
    fn rejects_generics_and_tuple_structs() {
        let err = expand("struct S<T> { #[position] clip: Vec4, t: T }").unwrap_err();
        assert!(err.contains("generic"), "err: {err}");
        let err = expand("struct S(Vec4);").unwrap_err();
        assert!(err.contains("named fields"), "err: {err}");
    }

    #[test]
    fn position_only_struct_is_legal() {
        // A varying-free interface: just the position. `VARYING_FIELDS` is
        // empty and the vertex writes only gl_Position through the literal.
        let out = expand("struct P { #[position] clip: Vec4 }").unwrap();
        assert!(
            out.contains("[(& 'static str , & 'static str) ; 0usize]"),
            "out: {out}"
        );
    }
}
