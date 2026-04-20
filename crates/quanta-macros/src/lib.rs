//! Proc macros for Quanta GPU kernels.

extern crate proc_macro;

mod compiler;
mod parse;
mod validate;

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{Expr, ItemFn, Lit, parse_macro_input};

/// Mark a function as a GPU kernel.
///
/// ```ignore
/// #[quanta::kernel]                  // default: O3
/// #[quanta::kernel(opt = "O2")]      // explicit O2
/// #[quanta::kernel(opt = "O0")]      // no optimization (debug)
/// ```
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    // Parse optimization level from attribute
    let opt_level = parse_opt_level(attr);

    if let Err(err) = validate::validate_kernel(&func) {
        return err.to_compile_error().into();
    }

    let mut kernel_def = match parse::parse_kernel(&func) {
        Ok(def) => def,
        Err(err) => return err.to_compile_error().into(),
    };
    kernel_def.opt_level = opt_level;

    let outputs = match compiler::compile_kernel(&kernel_def) {
        Ok(outputs) => outputs,
        Err(err) => {
            let msg = format!("quanta compiler error: {}", err);
            return syn::Error::new_spanned(&func.sig.ident, msg)
                .to_compile_error()
                .into();
        }
    };

    let func_name = &func.sig.ident;
    let binary_name = syn::Ident::new(
        &format!("{}_BINARY", func_name.to_string().to_uppercase()),
        func_name.span(),
    );

    let msl_expr = match &outputs.msl {
        Some(s) => {
            let s = s.as_str();
            quote! { Some(#s) }
        }
        None => quote! { None },
    };
    let wgsl_expr = match &outputs.wgsl {
        Some(s) => {
            let s = s.as_str();
            quote! { Some(#s) }
        }
        None => quote! { None },
    };
    let nvidia_expr = match &outputs.nvidia {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let amd_expr = match &outputs.amd {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let spirv_expr = match &outputs.spirv {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let metallib_expr = match &outputs.metallib {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };

    let expanded = quote! {
        pub static #binary_name: ::quanta::KernelBinary = ::quanta::KernelBinary {
            amd: #amd_expr,
            nvidia: #nvidia_expr,
            spirv: #spirv_expr,
            metallib: #metallib_expr,
            msl: #msl_expr,
            wgsl: #wgsl_expr,
            llvm_ir: None,
        };

        pub fn #func_name(device: &::quanta::Gpu) -> Result<::quanta::Wave, ::quanta::QuantaError> {
            let binary = #binary_name.for_vendor(device.caps().vendor)
                .ok_or_else(|| ::quanta::QuantaError::compilation_failed(
                    format!("no compiled kernel for vendor {:?}", device.caps().vendor)
                ))?;
            device.wave(binary)
        }
    };

    expanded.into()
}

/// Mark a function as a GPU device function (callable from kernels).
///
/// ```ignore
/// #[quanta::device]
/// fn activate(x: f32, threshold: f32) -> f32 {
///     if x > threshold { x } else { x * 0.99 }
/// }
/// ```
///
/// Device functions are inlined into kernels by LLVM.
/// They cannot be launched from CPU — only called from `#[quanta::kernel]` functions.
///
/// The proc macro captures the function source and emits a hidden constant
/// `__QUANTA_DEVICE_{NAME_UPPERCASE}` containing the source text. Kernel
/// compilation picks this up so MSL/WGSL emitters can prepend it as a regular
/// helper function.
#[proc_macro_attribute]
pub fn device(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
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

/// Mark a variable as shared (workgroup-local) memory inside a kernel.
///
/// ```ignore
/// #[quanta::kernel]
/// fn reduce(data: &[f32], result: &mut [f32]) {
///     #[quanta::shared] let local: [f32; 256];
///     let lid = local_id();
///     local[lid] = data[quark_id()];
///     barrier();
/// }
/// ```
///
/// When used inside a `#[quanta::kernel]` body, the kernel parser handles
/// this attribute directly — it emits `SharedDecl`, `SharedLoad`, and
/// `SharedStore` ops in the IR.
///
/// The proc macro itself is a no-op pass-through; the real work happens
/// in the kernel parser which inspects `let` attributes.
#[proc_macro_attribute]
pub fn shared(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// Mark a function as a vertex shader.
///
/// ```ignore
/// #[quanta::vertex]
/// fn transform(
///     pos: Vec3,
///     normal: Vec3,
///     mvp: &Mat4,
/// ) -> Vec4 {
///     mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
/// }
/// ```
///
/// Vertex shaders are paired with fragment shaders to form a render pipeline.
/// The macro captures the function source and emits a constant
/// `__QUANTA_VERTEX_{NAME_UPPERCASE}` for the driver to compile at pipeline
/// creation time.
#[proc_macro_attribute]
pub fn vertex(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    if matches!(func.sig.output, syn::ReturnType::Default) {
        return syn::Error::new_spanned(
            &func.sig.ident,
            "vertex shader must have a return type (clip-space position)",
        )
        .to_compile_error()
        .into();
    }

    let func_name = &func.sig.ident;
    let source = func.to_token_stream().to_string();
    let const_name = syn::Ident::new(
        &format!("__QUANTA_VERTEX_{}", func_name.to_string().to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub const #const_name: &str = #source;

        pub fn #func_name() -> &'static str {
            #source
        }
    };
    expanded.into()
}

/// Mark a function as a fragment shader.
///
/// ```ignore
/// #[quanta::fragment]
/// fn shade(
///     uv: Vec2,
///     color: Vec4,
///     albedo: &Texture2D,
/// ) -> Vec4 {
///     albedo.sample(uv) * color
/// }
/// ```
///
/// Fragment shaders receive interpolated outputs from the vertex stage and
/// produce a color per fragment. The macro captures the function source and
/// emits a constant `__QUANTA_FRAGMENT_{NAME_UPPERCASE}` for the driver to
/// compile at pipeline creation time.
#[proc_macro_attribute]
pub fn fragment(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    if matches!(func.sig.output, syn::ReturnType::Default) {
        return syn::Error::new_spanned(
            &func.sig.ident,
            "fragment shader must have a return type (output color)",
        )
        .to_compile_error()
        .into();
    }

    let func_name = &func.sig.ident;
    let source = func.to_token_stream().to_string();
    let const_name = syn::Ident::new(
        &format!("__QUANTA_FRAGMENT_{}", func_name.to_string().to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub const #const_name: &str = #source;

        pub fn #func_name() -> &'static str {
            #source
        }
    };
    expanded.into()
}

/// Parse `opt = "O2"` from the attribute.
/// Default: 3 (O3).
fn parse_opt_level(attr: TokenStream) -> u8 {
    if attr.is_empty() {
        return 3; // default O3
    }

    let parsed: Result<syn::MetaNameValue, _> = syn::parse(attr);
    if let Ok(nv) = parsed
        && nv.path.is_ident("opt")
        && let Expr::Lit(expr_lit) = &nv.value
        && let Lit::Str(s) = &expr_lit.lit
    {
        return match s.value().as_str() {
            "O0" | "0" => 0,
            "O1" | "1" => 1,
            "O2" | "2" => 2,
            "O3" | "3" => 3,
            _ => 3,
        };
    }
    3
}
