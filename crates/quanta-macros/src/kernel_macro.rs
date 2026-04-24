//! Implementation body for `#[quanta::kernel]`.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Expr, ItemFn, Lit, parse::Parser};

use crate::compiler;
use crate::parse;
use crate::validate;

/// Core implementation of the `#[quanta::kernel]` attribute macro.
pub(crate) fn expand_kernel(attr: TokenStream, func: ItemFn) -> TokenStream {
    let attr_str = attr.to_string();
    let is_jit = attr_str.contains("jit");
    let kernel_attrs = parse_kernel_attrs(attr.clone());

    if let Err(err) = validate::validate_kernel(&func) {
        return err.to_compile_error().into();
    }

    let mut kernel_def = match parse::parse_kernel(&func) {
        Ok(def) => def,
        Err(err) => return err.to_compile_error().into(),
    };
    kernel_def.opt_level = kernel_attrs.opt_level;
    kernel_def.workgroup_size = kernel_attrs.workgroup_size;
    kernel_def.subgroup_size = kernel_attrs.subgroup_size;

    if is_jit {
        return emit_jit_kernel(&func, &kernel_def);
    }

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
    let wgsl_expr = match &outputs.wgsl {
        Some(s) => quote! { Some(#s) },
        None => quote! { None },
    };

    let wg_x = kernel_attrs.workgroup_size[0];
    let wg_y = kernel_attrs.workgroup_size[1];
    let wg_z = kernel_attrs.workgroup_size[2];

    // Const generics: extract from the function signature and generate set_value calls
    let generics = &func.sig.generics;
    let mut const_setters = Vec::new();
    let num_regular_params = func.sig.inputs.len();
    for (i, generic) in func.sig.generics.params.iter().enumerate() {
        if let syn::GenericParam::Const(cp) = generic {
            let ident = &cp.ident;
            let slot = (num_regular_params + i) as u32;
            const_setters.push(quote! {
                wave.set_value(#slot, #ident as u32);
            });
        }
    }
    let const_generic_setters = quote! { #(#const_setters)* };

    let expanded = quote! {
        pub static #binary_name: ::quanta::KernelBinary = ::quanta::KernelBinary {
            amd: #amd_expr,
            nvidia: #nvidia_expr,
            spirv: #spirv_expr,
            metallib: #metallib_expr,
            wgsl: #wgsl_expr,
        };

        pub fn #func_name #generics (device: &::quanta::Gpu) -> Result<::quanta::Wave, ::quanta::QuantaError> {
            let binary = #binary_name.for_vendor(device.caps().vendor)
                .ok_or_else(|| ::quanta::QuantaError::compilation_failed(
                    format!("no compiled kernel for vendor {:?}", device.caps().vendor)
                ))?;
            let mut wave = device.wave(binary)?;
            wave.workgroup_size = [#wg_x, #wg_y, #wg_z];
            #const_generic_setters
            Ok(wave)
        }
    };

    expanded.into()
}

/// Emit JIT kernel: serialize KernelDef and embed it, generate runtime
/// compilation function via `wave_jit`.
fn emit_jit_kernel(func: &ItemFn, kernel_def: &quanta_ir::KernelDef) -> TokenStream {
    let func_name = &func.sig.ident;
    let def_name = syn::Ident::new(
        &format!("{}_DEF", func_name.to_string().to_uppercase()),
        func_name.span(),
    );

    let serialized = quanta_ir::serialize_kernel(kernel_def);
    let def_lit = proc_macro2::Literal::byte_string(&serialized);

    let wg_x = kernel_def.workgroup_size[0];
    let wg_y = kernel_def.workgroup_size[1];
    let wg_z = kernel_def.workgroup_size[2];

    let expanded = quote! {
        pub static #def_name: &[u8] = #def_lit;

        pub fn #func_name(device: &::quanta::Gpu) -> Result<::quanta::Wave, ::quanta::QuantaError> {
            let mut wave = device.wave_jit(#def_name)?;
            wave.workgroup_size = [#wg_x, #wg_y, #wg_z];
            Ok(wave)
        }
    };

    expanded.into()
}

/// Parsed kernel attributes from `#[quanta::kernel(...)]`.
struct KernelAttrs {
    opt_level: u8,
    workgroup_size: [u32; 3],
    subgroup_size: Option<u32>,
}

impl Default for KernelAttrs {
    fn default() -> Self {
        Self {
            opt_level: 3,
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
        }
    }
}

/// Parse kernel attributes: `opt = "O2"`, `workgroup = [16, 16, 1]`, `jit`.
///
/// Supports:
/// - `#[quanta::kernel]`                           -> defaults
/// - `#[quanta::kernel(opt = "O2")]`               -> opt only
/// - `#[quanta::kernel(workgroup = [256])]`        -> [256, 1, 1]
/// - `#[quanta::kernel(workgroup = [16, 16])]`     -> [16, 16, 1]
/// - `#[quanta::kernel(workgroup = [16, 16, 1])]`  -> explicit 3D
/// - `#[quanta::kernel(workgroup = [256], opt = "O2")]` -> both
fn parse_kernel_attrs(attr: TokenStream) -> KernelAttrs {
    let mut attrs = KernelAttrs::default();

    if attr.is_empty() {
        return attrs;
    }

    // Try parsing as a punctuated list of name = value pairs.
    // We use syn to parse the token stream as comma-separated meta items.
    let parser = syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated;
    let parsed = match parser.parse(attr.clone()) {
        Ok(p) => p,
        Err(_) => {
            // Fall back: might be just `jit` or a single `opt = "O2"`.
            // Try single MetaNameValue parse for backward compat.
            if let Ok(nv) = syn::parse::<syn::MetaNameValue>(attr)
                && nv.path.is_ident("opt")
                && let Expr::Lit(expr_lit) = &nv.value
                && let Lit::Str(s) = &expr_lit.lit
            {
                attrs.opt_level = parse_opt_str(&s.value());
            }
            return attrs;
        }
    };

    for meta in &parsed {
        match meta {
            syn::Meta::NameValue(nv) if nv.path.is_ident("opt") => {
                if let Expr::Lit(expr_lit) = &nv.value
                    && let Lit::Str(s) = &expr_lit.lit
                {
                    attrs.opt_level = parse_opt_str(&s.value());
                }
            }
            syn::Meta::NameValue(nv) if nv.path.is_ident("workgroup") => {
                if let Some(wg) = parse_workgroup_expr(&nv.value) {
                    attrs.workgroup_size = wg;
                }
            }
            syn::Meta::NameValue(nv) if nv.path.is_ident("subgroup") => {
                if let Expr::Lit(expr_lit) = &nv.value
                    && let Lit::Int(i) = &expr_lit.lit
                    && let Ok(v) = i.base10_parse::<u32>()
                {
                    attrs.subgroup_size = Some(v);
                }
            }
            _ => {} // ignore `jit` and unknown attrs
        }
    }

    attrs
}

fn parse_opt_str(s: &str) -> u8 {
    match s {
        "O0" | "0" => 0,
        "O1" | "1" => 1,
        "O2" | "2" => 2,
        "O3" | "3" => 3,
        _ => 3,
    }
}

/// Parse `[256]`, `[16, 16]`, or `[16, 16, 1]` from an expression.
fn parse_workgroup_expr(expr: &Expr) -> Option<[u32; 3]> {
    if let Expr::Array(arr) = expr {
        let elems: Vec<u32> = arr
            .elems
            .iter()
            .filter_map(|e| {
                if let Expr::Lit(lit) = e
                    && let Lit::Int(int_lit) = &lit.lit
                {
                    return int_lit.base10_parse::<u32>().ok();
                }
                None
            })
            .collect();
        match elems.len() {
            1 => return Some([elems[0], 1, 1]),
            2 => return Some([elems[0], elems[1], 1]),
            3 => return Some([elems[0], elems[1], elems[2]]),
            _ => {}
        }
    }
    None
}
