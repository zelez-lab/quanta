//! Implementation bodies for shader proc macros: vertex, fragment, tessellation,
//! mesh, and ray tracing stages.
//!
//! All nine stages emit through one builder, [`build_shader_binary`], so the
//! `ShaderBinary` literal — every field, including `wgsl` — is written in
//! exactly one place. Two families feed it: the *compiled* stages (vertex,
//! fragment) run the shader through the compiler binary; the *stub* stages
//! (tessellation, mesh, ray tracing) capture the entry point and emit an
//! all-`None` binary that the runtime fills in later.

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::ItemFn;

use quanta_dsl_core as compiler;

use crate::crate_path::CratePath;

/// The five backend-binary token expressions embedded in a `ShaderBinary`:
/// SPIR-V, the three metallib variants (macOS / iOS device / iOS simulator),
/// and WGSL. Each is a `proc_macro2::TokenStream` naming a `Some(..)`/`None`.
pub(crate) struct Backends {
    spirv: proc_macro2::TokenStream,
    metallib: proc_macro2::TokenStream,
    metallib_ios: proc_macro2::TokenStream,
    metallib_ios_sim: proc_macro2::TokenStream,
    wgsl: proc_macro2::TokenStream,
}

impl Backends {
    /// The all-`None` backends of a stub stage — no binaries embedded.
    fn none() -> Self {
        Backends {
            spirv: quote! { None },
            metallib: quote! { None },
            metallib_ios: quote! { None },
            metallib_ios_sim: quote! { None },
            wgsl: quote! { None },
        }
    }
}

/// The single place that writes the `ShaderBinary` literal — every field,
/// including `wgsl` — plus its `{NAME}_SHADER` static and the accessor fn.
///
/// `func_name` names both the accessor and (upper-cased + `_SHADER`) the
/// static; `stage` is the `::quanta::ShaderStage::<Variant>` path token; the
/// `backends` carry the five embedded-binary expressions.
pub(crate) fn build_shader_binary(
    func_name: &syn::Ident,
    stage: proc_macro2::TokenStream,
    backends: Backends,
    krate: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );
    let Backends {
        spirv,
        metallib,
        metallib_ios,
        metallib_ios_sim,
        wgsl,
    } = backends;

    let expanded = quote! {
        pub static #binary_name: #krate::ShaderBinary = #krate::ShaderBinary {
            spirv: #spirv,
            metallib: #metallib,
            metallib_ios: #metallib_ios,
            metallib_ios_sim: #metallib_ios_sim,
            wgsl: #wgsl,
            entry_point: #func_name_str,
            stage: #stage,
        };

        pub fn #func_name() -> &'static #krate::ShaderBinary {
            &#binary_name
        }
    };
    expanded
}

/// Embed compiled binary bytes as a `Some(&'static [u8])` byte-string
/// literal, or `None` when the variant wasn't produced. Shared by the
/// vertex/fragment paths for spirv + all three metallib variants.
fn embed_bytes(bytes: &Option<Vec<u8>>) -> proc_macro2::TokenStream {
    match bytes {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    }
}

/// Run the compiler binary over a parsed shader and produce the five
/// embedded-binary expressions. Shared by the direct compile path
/// ([`expand_compiled`]) and the second-stage varyings macros
/// (`varyings_macro`). The three compile outcomes: `Ok(Some)` embeds the
/// produced binaries, `Ok(None)` (no compiler on PATH) ships empty binaries
/// so `cargo check` works in fresh clones, and `Err` carries the message the
/// caller turns into a compile error.
pub(crate) fn compile_backends(
    func_name_str: &str,
    stage_str: &str,
    params: &[compiler::ShaderParam],
    return_ty: &compiler::ShaderType,
    body_source: &str,
    varyings: Option<&compiler::ShaderVaryings>,
) -> Result<Backends, String> {
    match compiler::compile_shader(
        func_name_str,
        stage_str,
        params,
        return_ty,
        body_source,
        varyings,
    ) {
        Ok(Some(output)) => Ok(Backends {
            spirv: embed_bytes(&output.spirv),
            metallib: embed_bytes(&output.metallib),
            metallib_ios: embed_bytes(&output.metallib_ios),
            metallib_ios_sim: embed_bytes(&output.metallib_ios_sim),
            wgsl: match &output.wgsl {
                Some(s) => quote! { Some(#s) },
                None => quote! { None },
            },
        }),
        Ok(None) => Ok(Backends::none()),
        Err(msg) => Err(msg),
    }
}

/// Shared body of the *compiled* stages (vertex, fragment): parse params +
/// return type, run the compiler binary, and emit through
/// [`build_shader_binary`]. `stage_str` is the compiler's stage name and
/// `stage` the matching `ShaderStage` path token; `allow_textures` gates the
/// `&Texture2D` params that only fragment shaders accept.
///
/// The three compile outcomes match the original vertex/fragment code exactly:
/// `Ok(Some)` embeds the produced binaries, `Ok(None)` (no compiler on PATH)
/// ships empty binaries so `cargo check` works in fresh clones, and `Err`
/// fails the build rather than deferring a panic to pipeline creation.
fn expand_compiled(
    func: ItemFn,
    stage_str: &str,
    stage: proc_macro2::TokenStream,
    allow_textures: bool,
    krate: &proc_macro2::TokenStream,
) -> TokenStream {
    if matches!(func.sig.output, syn::ReturnType::Default) {
        let msg = if allow_textures {
            "fragment shader must have a return type (output color)"
        } else {
            "vertex shader must have a return type (clip-space position)"
        };
        return syn::Error::new_spanned(&func.sig.ident, msg)
            .to_compile_error()
            .into();
    }

    let func_name = func.sig.ident.clone();
    let func_name_str = func_name.to_string();

    // Parse shader params and body, then compile via the compiler binary.
    let (params, textures) = match compiler::parse_shader_params(&func) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };
    if !allow_textures && !textures.is_empty() {
        return syn::Error::new_spanned(
            &func.sig.ident,
            "texture parameters are only supported in fragment shaders",
        )
        .to_compile_error()
        .into();
    }
    let return_ty = match compiler::parse_return_type(&func) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    // Vertex ships the body verbatim; fragment rewrites `&Texture2D` params
    // to slots by declaration order, since the emitters consume the slot
    // form (`sample(N, uv)`).
    let body_source = if allow_textures {
        compiler::rewrite_texture_names(&func.block.to_token_stream().to_string(), &textures)
    } else {
        func.block.to_token_stream().to_string()
    };

    let backends = match compile_backends(
        &func_name_str,
        stage_str,
        &params,
        &return_ty,
        &body_source,
        None,
    ) {
        Ok(b) => b,
        // Compiler found but failed — a shader with missing binaries
        // panics at pipeline creation, so fail the build here instead.
        Err(msg) => {
            return syn::Error::new_spanned(
                &func.sig.ident,
                format!("{stage_str} shader `{func_name_str}` failed to compile: {msg}"),
            )
            .to_compile_error()
            .into();
        }
    };

    build_shader_binary(&func_name, stage, backends, krate).into()
}

/// Shared body of the *stub* stages (tessellation, mesh, ray tracing): capture
/// the entry point and emit an all-`None` [`ShaderBinary`] through
/// [`build_shader_binary`]. These stages don't run the compiler binary; the
/// runtime fills the binaries in later.
fn expand_stub(
    func: ItemFn,
    stage: proc_macro2::TokenStream,
    krate: &proc_macro2::TokenStream,
) -> TokenStream {
    build_shader_binary(&func.sig.ident, stage, Backends::none(), krate).into()
}

/// The shader-primitive type NAMES the DSL knows in signatures. A vertex
/// returning one of these (in practice `Vec4`) is a position-only vertex; a
/// return type OUTSIDE this set names a `#[derive(Varyings)]` struct and
/// routes through its trampoline. The same test classifies a fragment param
/// as the varyings receiver.
const SHADER_PRIMITIVES: &[&str] = &["f32", "u32", "Vec2", "Vec3", "Vec4", "Mat3", "Mat4"];

/// The last path-segment name of a bare (non-reference) path type, if any.
fn bare_type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

/// Route an item through a Varyings struct's trampoline: expands to
/// `__quanta_varyings_<Name>! { (<macros_root>::<callback>) (<attr>) <fn> }`
/// — the derive-generated macro pastes the struct's field metadata in front
/// and forwards everything to the hidden second-stage proc-macro, which has
/// BOTH the interface and the function and runs the real compile. The
/// trampoline resolves like any macro_rules name: the struct must be
/// declared before the shader, or its `__quanta_varyings_<Name>` re-export
/// imported alongside it.
fn expand_via_trampoline(
    struct_name: &str,
    callback: proc_macro2::TokenStream,
    attr: proc_macro2::TokenStream,
    func: &ItemFn,
) -> TokenStream {
    let mac = syn::Ident::new(
        &format!("__quanta_varyings_{struct_name}"),
        proc_macro2::Span::call_site(),
    );
    quote! {
        #mac! { (#callback) (#attr) #func }
    }
    .into()
}

/// Core implementation of `#[quanta::vertex]`.
///
/// Dispatch: a vertex returning a shader primitive (`-> Vec4`) is the
/// position-only form and compiles directly; a vertex returning any other
/// bare type name (`-> Surface`) uses the shared-struct varying model and
/// routes through the struct's Varyings trampoline to `__vertex_varyings`.
pub(crate) fn expand_vertex(
    func: ItemFn,
    attr: proc_macro2::TokenStream,
    cp: &CratePath,
) -> TokenStream {
    if let syn::ReturnType::Type(_, ty) = &func.sig.output
        && let Some(name) = bare_type_name(ty)
        && !SHADER_PRIMITIVES.contains(&name.as_str())
    {
        let macros_root = cp.macros();
        return expand_via_trampoline(
            &name,
            quote! { #macros_root::__vertex_varyings },
            attr,
            &func,
        );
    }
    let krate = cp.types();
    expand_compiled(
        func,
        "vertex",
        quote! { #krate::ShaderStage::Vertex },
        false,
        &krate,
    )
}

/// Core implementation of `#[quanta::fragment]`.
///
/// Dispatch: a fragment whose params include one typed as a non-primitive
/// bare name (`s: Surface`) consumes the shared-struct varying interface and
/// routes through the struct's Varyings trampoline to `__fragment_varyings`.
/// A fragment with a plain shader-primitive VALUE param is an error — the
/// positional varying model is gone; varyings come from the struct.
pub(crate) fn expand_fragment(
    func: ItemFn,
    attr: proc_macro2::TokenStream,
    cp: &CratePath,
) -> TokenStream {
    let mut receiver_struct: Option<String> = None;
    for arg in &func.sig.inputs {
        let syn::FnArg::Typed(pat_type) = arg else {
            continue;
        };
        let Some(name) = bare_type_name(&pat_type.ty) else {
            continue; // references (&T uniforms, &[T] slices, &Texture2D)
        };
        if SHADER_PRIMITIVES.contains(&name.as_str()) {
            return syn::Error::new_spanned(
                &pat_type.ty,
                "fragment stage inputs come from a #[derive(Varyings)] struct: declare the \
                 varyings there, take the struct as a single param (`s: Surface`), and read \
                 them as `s.<field>` — plain value params are vertex-only attributes",
            )
            .to_compile_error()
            .into();
        }
        if name == "Texture2D" {
            continue; // by-value texture error surfaces in param parsing
        }
        if let Some(first) = &receiver_struct {
            return syn::Error::new_spanned(
                &pat_type.ty,
                format!(
                    "a fragment takes at most one Varyings struct param (already \
                     receiving `{first}`)"
                ),
            )
            .to_compile_error()
            .into();
        }
        receiver_struct = Some(name);
    }
    if let Some(struct_name) = receiver_struct {
        let macros_root = cp.macros();
        return expand_via_trampoline(
            &struct_name,
            quote! { #macros_root::__fragment_varyings },
            attr,
            &func,
        );
    }
    let krate = cp.types();
    expand_compiled(
        func,
        "fragment",
        quote! { #krate::ShaderStage::Fragment },
        true,
        &krate,
    )
}

/// Core implementation of `#[quanta::tess_control]`.
pub(crate) fn expand_tess_control(func: ItemFn, cp: &CratePath) -> TokenStream {
    let krate = cp.types();
    expand_stub(func, quote! { #krate::ShaderStage::TessControl }, &krate)
}

/// Core implementation of `#[quanta::tess_eval]`.
pub(crate) fn expand_tess_eval(func: ItemFn, cp: &CratePath) -> TokenStream {
    let krate = cp.types();
    expand_stub(func, quote! { #krate::ShaderStage::TessEval }, &krate)
}

/// Core implementation of `#[quanta::task]`.
pub(crate) fn expand_task(func: ItemFn, cp: &CratePath) -> TokenStream {
    let krate = cp.types();
    expand_stub(func, quote! { #krate::ShaderStage::Task }, &krate)
}

/// Core implementation of `#[quanta::mesh]`.
pub(crate) fn expand_mesh(func: ItemFn, cp: &CratePath) -> TokenStream {
    let krate = cp.types();
    expand_stub(func, quote! { #krate::ShaderStage::Mesh }, &krate)
}

/// Core implementation of `#[quanta::ray_gen]`.
pub(crate) fn expand_ray_gen(func: ItemFn, cp: &CratePath) -> TokenStream {
    let krate = cp.types();
    expand_stub(func, quote! { #krate::ShaderStage::RayGen }, &krate)
}

/// Core implementation of `#[quanta::closest_hit]`.
pub(crate) fn expand_closest_hit(func: ItemFn, cp: &CratePath) -> TokenStream {
    let krate = cp.types();
    expand_stub(func, quote! { #krate::ShaderStage::ClosestHit }, &krate)
}

/// Core implementation of `#[quanta::miss]`.
pub(crate) fn expand_miss(func: ItemFn, cp: &CratePath) -> TokenStream {
    let krate = cp.types();
    expand_stub(func, quote! { #krate::ShaderStage::Miss }, &krate)
}
