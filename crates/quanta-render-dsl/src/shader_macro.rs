//! Implementation bodies for shader proc macros: vertex, fragment, tessellation,
//! mesh, and ray tracing stages.

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::ItemFn;

use quanta_dsl_core as compiler;

/// Core implementation of `#[quanta::vertex]`.
pub(crate) fn expand_vertex(func: ItemFn) -> TokenStream {
    if matches!(func.sig.output, syn::ReturnType::Default) {
        return syn::Error::new_spanned(
            &func.sig.ident,
            "vertex shader must have a return type (clip-space position)",
        )
        .to_compile_error()
        .into();
    }

    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    // Parse shader params and body, then compile via the compiler binary.
    let (params, textures) = match compiler::parse_shader_params(&func) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };
    if !textures.is_empty() {
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

    // Extract body source text for the compiler.
    let body_source = func.block.to_token_stream().to_string();

    let (spirv_expr, metallib_expr, wgsl_expr) =
        match compiler::compile_shader(&func_name_str, "vertex", &params, &return_ty, &body_source)
        {
            Ok(Some(output)) => {
                let spirv = match &output.spirv {
                    Some(bytes) => {
                        let lit = proc_macro2::Literal::byte_string(bytes);
                        quote! { Some(#lit as &[u8]) }
                    }
                    None => quote! { None },
                };
                let metallib = match &output.metallib {
                    Some(bytes) => {
                        let lit = proc_macro2::Literal::byte_string(bytes);
                        quote! { Some(#lit as &[u8]) }
                    }
                    None => quote! { None },
                };
                let wgsl = match &output.wgsl {
                    Some(s) => quote! { Some(#s) },
                    None => quote! { None },
                };
                (spirv, metallib, wgsl)
            }
            // No compiler binary found — ship empty binaries so `cargo
            // check` works in fresh clones; the runtime reports the gap.
            Ok(None) => (quote! { None }, quote! { None }, quote! { None }),
            // Compiler found but failed — a shader with missing binaries
            // panics at pipeline creation, so fail the build here instead.
            Err(msg) => {
                return syn::Error::new_spanned(
                    &func.sig.ident,
                    format!("vertex shader `{func_name_str}` failed to compile: {msg}"),
                )
                .to_compile_error()
                .into();
            }
        };

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: #spirv_expr,
            metallib: #metallib_expr,
            wgsl: #wgsl_expr,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Vertex,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Core implementation of `#[quanta::fragment]`.
pub(crate) fn expand_fragment(func: ItemFn) -> TokenStream {
    if matches!(func.sig.output, syn::ReturnType::Default) {
        return syn::Error::new_spanned(
            &func.sig.ident,
            "fragment shader must have a return type (output color)",
        )
        .to_compile_error()
        .into();
    }

    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    // Parse shader params and body, then compile via the compiler binary.
    let (params, textures) = match compiler::parse_shader_params(&func) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };
    let return_ty = match compiler::parse_return_type(&func) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    // `&Texture2D` params resolve to slots by declaration order; the
    // emitters consume the slot form (`sample(N, uv)`).
    let body_source =
        compiler::rewrite_texture_names(&func.block.to_token_stream().to_string(), &textures);

    let (spirv_expr, metallib_expr, wgsl_expr) = match compiler::compile_shader(
        &func_name_str,
        "fragment",
        &params,
        &return_ty,
        &body_source,
    ) {
        Ok(Some(output)) => {
            let spirv = match &output.spirv {
                Some(bytes) => {
                    let lit = proc_macro2::Literal::byte_string(bytes);
                    quote! { Some(#lit as &[u8]) }
                }
                None => quote! { None },
            };
            let metallib = match &output.metallib {
                Some(bytes) => {
                    let lit = proc_macro2::Literal::byte_string(bytes);
                    quote! { Some(#lit as &[u8]) }
                }
                None => quote! { None },
            };
            let wgsl = match &output.wgsl {
                Some(s) => quote! { Some(#s) },
                None => quote! { None },
            };
            (spirv, metallib, wgsl)
        }
        // No compiler binary found — ship empty binaries so `cargo
        // check` works in fresh clones; the runtime reports the gap.
        Ok(None) => (quote! { None }, quote! { None }, quote! { None }),
        // Compiler found but failed — a shader with missing binaries
        // panics at pipeline creation, so fail the build here instead.
        Err(msg) => {
            return syn::Error::new_spanned(
                &func.sig.ident,
                format!("fragment shader `{func_name_str}` failed to compile: {msg}"),
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: #spirv_expr,
            metallib: #metallib_expr,
            wgsl: #wgsl_expr,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Fragment,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Core implementation of `#[quanta::tess_control]`.
pub(crate) fn expand_tess_control(func: ItemFn) -> TokenStream {
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::TessControl,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Core implementation of `#[quanta::tess_eval]`.
pub(crate) fn expand_tess_eval(func: ItemFn) -> TokenStream {
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::TessEval,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Core implementation of `#[quanta::task]`.
pub(crate) fn expand_task(func: ItemFn) -> TokenStream {
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Task,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Core implementation of `#[quanta::mesh]`.
pub(crate) fn expand_mesh(func: ItemFn) -> TokenStream {
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Mesh,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Core implementation of `#[quanta::ray_gen]`.
pub(crate) fn expand_ray_gen(func: ItemFn) -> TokenStream {
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::RayGen,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Core implementation of `#[quanta::closest_hit]`.
pub(crate) fn expand_closest_hit(func: ItemFn) -> TokenStream {
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::ClosestHit,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Core implementation of `#[quanta::miss]`.
pub(crate) fn expand_miss(func: ItemFn) -> TokenStream {
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Miss,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}
