//! Second-stage expansion for shaders that use the shared-struct varying
//! model (`#[derive(Varyings)]`).
//!
//! `#[quanta::vertex]` / `#[quanta::fragment]` cannot see the Varyings
//! struct's fields (a proc macro sees only its own item), so they expand to
//! an invocation of the struct's derive-generated trampoline
//! (`__quanta_varyings_<Name>!`), which pastes the interface metadata in
//! front of the function tokens and forwards the whole package here — to the
//! hidden `__vertex_varyings` / `__fragment_varyings` proc-macros. This
//! module parses that package, builds the `ShaderVaryings` interface, runs
//! the shader compiler, and emits the same `ShaderBinary` static the direct
//! path emits.
//!
//! Wire format of the package (produced by the trampoline):
//!
//! ```text
//! @varyings Surface { #[position] clip : Vec4 , uv : Vec2 , kind : u32 }
//! (<original attribute args>)
//! fn vs(...) -> Surface { ... }
//! ```

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, ItemFn, Token, braced, parenthesized};

use quanta_dsl_core as compiler;

use crate::shader_macro::{build_shader_binary, compile_backends};

/// One field of the trampoline's interface metadata: `[#[position]] name : Ty`.
struct MetaField {
    is_position: bool,
    name: Ident,
    ty: Ident,
}

impl Parse for MetaField {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(syn::Attribute::parse_outer)?;
        let is_position = attrs.iter().any(|a| a.path().is_ident("position"));
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty: Ident = input.parse()?;
        Ok(MetaField {
            is_position,
            name,
            ty,
        })
    }
}

/// The full package a trampoline forwards: the interface metadata, the
/// original attribute args, and the shader function.
pub(crate) struct VaryingsCall {
    struct_name: Ident,
    fields: Vec<MetaField>,
    attr: TokenStream2,
    func: ItemFn,
}

impl Parse for VaryingsCall {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![@]>()?;
        let kw: Ident = input.parse()?;
        if kw != "varyings" {
            return Err(syn::Error::new_spanned(kw, "expected `@varyings`"));
        }
        let struct_name: Ident = input.parse()?;
        let content;
        braced!(content in input);
        let fields: syn::punctuated::Punctuated<MetaField, Token![,]> =
            content.parse_terminated(MetaField::parse, Token![,])?;
        let attr_content;
        parenthesized!(attr_content in input);
        let attr: TokenStream2 = attr_content.parse()?;
        let func: ItemFn = input.parse()?;
        Ok(VaryingsCall {
            struct_name,
            fields: fields.into_iter().collect(),
            attr,
            func,
        })
    }
}

/// Map a metadata type name to the DSL's `ShaderType`. The derive validated
/// the set already; an unknown name here means the trampoline was edited by
/// hand.
fn shader_type(ty: &Ident) -> syn::Result<compiler::ShaderType> {
    compiler::shader_types::shader_type_from_ident(&ty.to_string())
        .map_err(|msg| syn::Error::new_spanned(ty, msg))
}

/// Build the `ShaderVaryings` interface from the trampoline metadata.
/// `binding` is the fragment's receiver param name (`None` on the vertex).
fn build_varyings(
    call: &VaryingsCall,
    binding: Option<String>,
) -> syn::Result<compiler::ShaderVaryings> {
    let mut position: Option<String> = None;
    let mut fields: Vec<(String, compiler::ShaderType)> = Vec::new();
    for f in &call.fields {
        if f.is_position {
            if position.is_some() {
                return Err(syn::Error::new_spanned(
                    &f.name,
                    "duplicate #[position] field in the Varyings interface",
                ));
            }
            position = Some(f.name.to_string());
        } else {
            fields.push((f.name.to_string(), shader_type(&f.ty)?));
        }
    }
    let position = position.ok_or_else(|| {
        syn::Error::new_spanned(
            &call.struct_name,
            "the Varyings interface names no #[position] field",
        )
    })?;
    Ok(compiler::ShaderVaryings {
        struct_name: call.struct_name.to_string(),
        position,
        fields,
        binding,
    })
}

/// The `#[quanta::vertex]` second stage: the function RETURNS the Varyings
/// struct; its body ends in the struct literal. Params are ordinary vertex
/// attributes / uniforms / slices (no textures in a vertex).
pub(crate) fn expand_vertex_varyings(input: TokenStream) -> TokenStream {
    expand2(input.into(), Stage::Vertex)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// The `#[quanta::fragment]` second stage: the function TAKES the Varyings
/// struct as its single stage-input param (the receiver); the body reads
/// varyings as `<receiver>.<field>`. Other params are uniforms / slices /
/// textures — a plain value param is rejected.
pub(crate) fn expand_fragment_varyings(input: TokenStream) -> TokenStream {
    expand2(input.into(), Stage::Fragment)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

enum Stage {
    Vertex,
    Fragment,
}

/// `proc_macro2` core of both entries — parse the trampoline package, then
/// expand. Split from the `proc_macro` wrappers so unit tests can drive it.
fn expand2(input: TokenStream2, stage: Stage) -> syn::Result<TokenStream2> {
    let call: VaryingsCall = syn::parse2(input)?;
    expand_stage(&call, stage)
}

fn expand_stage(call: &VaryingsCall, stage: Stage) -> syn::Result<TokenStream2> {
    let cp = crate::crate_path::from_attr_args2(call.attr.clone());
    let krate = cp.types();
    let func = &call.func;
    let func_name = func.sig.ident.clone();
    let func_name_str = func_name.to_string();

    let (stage_str, stage_path, varyings, params, textures, mut body_source) = match stage {
        Stage::Vertex => {
            // The return type must name the trampoline's struct (the stage-1
            // dispatch derived the trampoline name from it, so a mismatch
            // means a hand-written invocation).
            let ret_ok = matches!(
                &func.sig.output,
                syn::ReturnType::Type(_, ty)
                    if matches!(ty.as_ref(), syn::Type::Path(p)
                        if p.path.segments.last().is_some_and(|s| s.ident == call.struct_name))
            );
            if !ret_ok {
                return Err(syn::Error::new_spanned(
                    &func.sig.output,
                    format!(
                        "a varyings vertex must return the interface struct `{}`",
                        call.struct_name
                    ),
                ));
            }
            let (params, textures) = compiler::parse_shader_params(func)?;
            if !textures.is_empty() {
                return Err(syn::Error::new_spanned(
                    &func.sig.ident,
                    "texture parameters are only supported in fragment shaders",
                ));
            }
            let varyings = build_varyings(call, None)?;
            let body = func.block.to_token_stream().to_string();
            (
                "vertex",
                quote! { #krate::ShaderStage::Vertex },
                varyings,
                params,
                textures,
                body,
            )
        }
        Stage::Fragment => {
            // Split the receiver param (typed as the interface struct) out of
            // the signature; everything else parses as ordinary shader params.
            let mut receiver: Option<String> = None;
            let mut stripped = func.clone();
            stripped.sig.inputs = func
                .sig
                .inputs
                .iter()
                .filter(|arg| {
                    if let syn::FnArg::Typed(pat_type) = arg
                        && let syn::Type::Path(p) = pat_type.ty.as_ref()
                        && p.path
                            .segments
                            .last()
                            .is_some_and(|s| s.ident == call.struct_name)
                    {
                        if let syn::Pat::Ident(pi) = pat_type.pat.as_ref() {
                            receiver = Some(pi.ident.to_string());
                        }
                        return false;
                    }
                    true
                })
                .cloned()
                .collect();
            let receiver = receiver.ok_or_else(|| {
                syn::Error::new_spanned(
                    &func.sig.ident,
                    format!(
                        "a varyings fragment must take the interface struct `{}` as a param",
                        call.struct_name
                    ),
                )
            })?;
            let (params, textures) = compiler::parse_shader_params(&stripped)?;
            // Everything that reached `params` as a plain value is illegal
            // here — fragment stage inputs come from the struct.
            if let Some(p) = params.iter().find(|p| !p.is_uniform && !p.is_slice) {
                return Err(syn::Error::new_spanned(
                    &func.sig.ident,
                    format!(
                        "fragment param `{}`: fragment stage inputs come from the \
                         #[derive(Varyings)] struct — declare it as a field and read \
                         `{receiver}.{}`",
                        p.name, p.name
                    ),
                ));
            }
            let varyings = build_varyings(call, Some(receiver))?;
            let body = func.block.to_token_stream().to_string();
            (
                "fragment",
                quote! { #krate::ShaderStage::Fragment },
                varyings,
                params,
                textures,
                body,
            )
        }
    };

    // Fragments rewrite `sample(name, uv)` to the slot form the emitters
    // bind; vertices carry no textures (checked above), so the rewrite is a
    // no-op there.
    if !textures.is_empty() {
        body_source = compiler::rewrite_texture_names(&body_source, &textures);
    }

    // The Varyings struct itself is the interface descriptor; the ShaderDef
    // return type stays the position's Vec4 (what gl_Position carries).
    let return_ty = compiler::ShaderType::Vec4;
    let backends = compile_backends(
        &func_name_str,
        stage_str,
        &params,
        &return_ty,
        &body_source,
        Some(&varyings),
    )
    .map_err(|msg| {
        syn::Error::new_spanned(
            &func.sig.ident,
            format!("{stage_str} shader `{func_name_str}` failed to compile: {msg}"),
        )
    })?;

    Ok(build_shader_binary(
        &func_name, stage_path, backends, &krate,
    ))
}

#[cfg(test)]
mod tests {
    use super::{Stage, expand2};
    use quote::quote;

    /// Pin `QUANTA_COMPILER` to this workspace's own build of the compiler
    /// (target/debug or target/release), so the end-to-end tests are
    /// hermetic: a developer environment may point `QUANTA_COMPILER` at a
    /// pinned external compiler whose rev differs, and the stale-compiler
    /// handshake would (correctly) hard-error. When neither profile has
    /// built the binary yet, the variable is cleared so resolution falls
    /// through to "no compiler" and the macros still expand (with `None`
    /// binaries).
    fn pin_workspace_compiler() {
        let ws = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let candidate = [
            "target/debug/quanta-compiler",
            "target/release/quanta-compiler",
        ]
        .iter()
        .map(|p| ws.join(p))
        .find(|p| p.exists());
        // SAFETY: test-process env mutation; the only concurrent writers are
        // this module's tests, which all write the same value.
        unsafe {
            match candidate {
                Some(p) => std::env::set_var("QUANTA_COMPILER", p),
                None => std::env::remove_var("QUANTA_COMPILER"),
            }
        }
    }

    /// The flagship pair's trampoline package, as the derive-generated
    /// trampoline would paste it (vertex side).
    fn vertex_pkg() -> proc_macro2::TokenStream {
        quote! {
            @varyings Surface { #[position] clip : Vec4 , uv : Vec2 , kind : u32 }
            ()
            fn vs(pos: Vec3, in_uv: Vec2) -> Surface {
                Surface {
                    clip: Vec4::new(pos.x, pos.y, 0.0, 1.0),
                    uv: in_uv,
                    kind: 0u32,
                }
            }
        }
    }

    fn fragment_pkg() -> proc_macro2::TokenStream {
        quote! {
            @varyings Surface { #[position] clip : Vec4 , uv : Vec2 , kind : u32 }
            ()
            fn fs(s: Surface) -> Vec4 {
                let c = if s.kind == 1u32 { 1.0 } else { 0.2 };
                Vec4::new(s.uv.x, s.uv.y, c, 1.0)
            }
        }
    }

    /// The full second stage over the REAL compiler binary (when present in
    /// target/): the vertex package expands to the `VS_SHADER` static and
    /// accessor — the same surface the direct path emits. Without a compiler
    /// on PATH the static still emits (with `None` binaries), so this holds
    /// on fresh clones too.
    #[test]
    fn vertex_package_expands_to_shader_binary_static() {
        pin_workspace_compiler();
        let out = expand2(vertex_pkg(), Stage::Vertex).unwrap().to_string();
        assert!(out.contains("VS_SHADER"), "out: {out}");
        assert!(out.contains("ShaderStage :: Vertex"), "out: {out}");
        assert!(out.contains("pub fn vs ()"), "out: {out}");
    }

    #[test]
    fn fragment_package_expands_to_shader_binary_static() {
        pin_workspace_compiler();
        let out = expand2(fragment_pkg(), Stage::Fragment)
            .unwrap()
            .to_string();
        assert!(out.contains("FS_SHADER"), "out: {out}");
        assert!(out.contains("ShaderStage :: Fragment"), "out: {out}");
        assert!(out.contains("pub fn fs ()"), "out: {out}");
    }

    #[test]
    fn vertex_must_return_the_interface_struct() {
        let pkg = quote! {
            @varyings Surface { #[position] clip : Vec4 , uv : Vec2 }
            ()
            fn vs(pos: Vec3) -> Vec4 { Vec4::new(pos.x, pos.y, 0.0, 1.0) }
        };
        let err = expand2(pkg, Stage::Vertex).unwrap_err().to_string();
        assert!(
            err.contains("must return the interface struct"),
            "err: {err}"
        );
    }

    #[test]
    fn fragment_needs_the_receiver_param() {
        let pkg = quote! {
            @varyings Surface { #[position] clip : Vec4 , uv : Vec2 }
            ()
            fn fs() -> Vec4 { Vec4::new(1.0, 0.0, 0.0, 1.0) }
        };
        let err = expand2(pkg, Stage::Fragment).unwrap_err().to_string();
        assert!(err.contains("must take the interface struct"), "err: {err}");
    }

    #[test]
    fn fragment_rejects_plain_value_params() {
        let pkg = quote! {
            @varyings Surface { #[position] clip : Vec4 , uv : Vec2 }
            ()
            fn fs(s: Surface, extra: Vec2) -> Vec4 { Vec4::new(extra.x, 0.0, 0.0, 1.0) }
        };
        let err = expand2(pkg, Stage::Fragment).unwrap_err().to_string();
        assert!(
            err.contains("come from the #[derive(Varyings)] struct"),
            "err: {err}"
        );
    }

    #[test]
    fn vertex_rejects_texture_params() {
        let pkg = quote! {
            @varyings Surface { #[position] clip : Vec4 }
            ()
            fn vs(pos: Vec3, tex: &Texture2D) -> Surface {
                Surface { clip: Vec4::new(pos.x, pos.y, 0.0, 1.0) }
            }
        };
        let err = expand2(pkg, Stage::Vertex).unwrap_err().to_string();
        assert!(err.contains("only supported in fragment"), "err: {err}");
    }

    #[test]
    fn interface_without_position_is_rejected() {
        let pkg = quote! {
            @varyings Surface { uv : Vec2 }
            ()
            fn vs(pos: Vec3) -> Surface { Surface { uv: pos } }
        };
        let err = expand2(pkg, Stage::Vertex).unwrap_err().to_string();
        assert!(err.contains("no #[position] field"), "err: {err}");
    }
}
