#![allow(dead_code)]
//! Auto-dispatch code generation for struct-ref kernel parameters.
//!
//! When `#[quanta::kernel]` detects a parameter typed as `p: &MyStruct` where
//! MyStruct derives `quanta::Fields`, it generates:
//!
//! 1. `fn kernel_wave(gpu)` — the existing wave-creation function
//! 2. `fn kernel(gpu, data, quarks)` — auto-dispatch that handles
//!    alloc → upload → bind → dispatch → readback
//!
//! Field access patterns are discovered by scanning the kernel body for
//! `p.field_name[idx]` (buffer access) and `p.field_name` (scalar/push constant)
//! expressions, then classifying each as read-only, write-only, or read-write.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ItemFn;

/// Discovered field access pattern on a struct-ref kernel parameter.
pub(crate) struct StructFieldAccess {
    /// Field name (e.g., "pos_x").
    pub name: String,
    /// Slot index for GPU binding.
    pub slot: usize,
    /// True if accessed with indexing (p.field[idx]) — buffer field.
    pub is_indexed: bool,
    /// True if the kernel reads this field.
    pub is_read: bool,
    /// True if the kernel writes this field.
    pub is_written: bool,
    /// Scalar type name (e.g., "f32").
    pub scalar_type_name: String,
}

/// Metadata about the struct-ref parameter for code generation.
pub(crate) struct StructParamInfo {
    /// The parameter name (e.g., "p" or "data").
    pub param_name: String,
    /// The type name (e.g., "Particles").
    pub type_name: String,
    /// The type path tokens (for use in generated code).
    pub type_tokens: TokenStream,
    /// Discovered field accesses from the kernel body.
    pub fields: Vec<StructFieldAccess>,
}

/// Generate the auto-dispatch function for a struct-ref kernel.
///
/// The generated function signature is:
/// ```ignore
/// pub fn kernel_name(
///     device: &::quanta::Gpu,
///     data: &mut StructType,
///     quarks: u32,
/// ) -> Result<::quanta::Pulse, ::quanta::QuantaError>
/// ```
pub(crate) fn emit_auto_dispatch(
    func: &ItemFn,
    info: &StructParamInfo,
    wave_fn_name: &syn::Ident,
) -> TokenStream {
    let func_name = &func.sig.ident;
    let param_ident = format_ident!("{}", info.param_name);
    let type_tokens = &info.type_tokens;

    // Separate buffer fields (Vec<T>) from scalar fields (push constants).
    // Buffer fields are those accessed with indexing: p.field[idx]
    // Scalar fields are those accessed without indexing: p.field
    let buffer_fields: Vec<_> = info.fields.iter().filter(|f| f.is_indexed).collect();
    let scalar_fields: Vec<_> = info.fields.iter().filter(|f| !f.is_indexed).collect();

    // Generate field allocation + upload for each buffer field
    let mut alloc_stmts = Vec::new();
    let mut upload_stmts = Vec::new();
    let mut bind_stmts = Vec::new();
    let mut readback_stmts = Vec::new();

    for (i, field) in buffer_fields.iter().enumerate() {
        let field_var = format_ident!("__f{}", i);
        let field_ident = format_ident!("{}", field.name);
        let slot = field.slot as u32;
        let scalar_ty = scalar_type_to_rust_tokens(field.scalar_type_name.as_str());

        // Allocate
        alloc_stmts.push(quote! {
            let #field_var = device.compute_field::<#scalar_ty>(#param_ident.#field_ident.len())?;
        });

        // Upload (only if field is read by the kernel)
        if field.is_read {
            upload_stmts.push(quote! {
                #field_var.write(&#param_ident.#field_ident)?;
            });
        }

        // Bind
        bind_stmts.push(quote! {
            __wave.bind(#slot, &#field_var);
        });

        // Readback (only if field is written by the kernel)
        if field.is_written {
            readback_stmts.push(quote! {
                #param_ident.#field_ident = #field_var.read()?;
            });
        }
    }

    // Generate push constant setters for scalar fields
    let mut push_stmts = Vec::new();
    for field in &scalar_fields {
        let field_ident = format_ident!("{}", field.name);
        let slot = field.slot as u32;

        push_stmts.push(quote! {
            __wave.set_value(#slot, #param_ident.#field_ident);
        });
    }

    // Const generics from the original function: forward them as parameters
    // and generate set_value calls for them
    let mut const_params = Vec::new();
    let mut const_setters = Vec::new();
    let total_field_count = info.fields.len();
    for (i, generic) in func.sig.generics.params.iter().enumerate() {
        if let syn::GenericParam::Const(cp) = generic {
            let ident = &cp.ident;
            let ty = &cp.ty;
            let slot = (total_field_count + i) as u32;
            const_params.push(quote! { #ident: #ty });
            const_setters.push(quote! {
                __wave.set_value(#slot, #ident as u32);
            });
        }
    }

    // Build the full generics for const generic forwarding
    let generics = &func.sig.generics;

    let expanded = quote! {
        pub fn #func_name #generics (
            device: &::quanta::Gpu,
            #param_ident: &mut #type_tokens,
            quarks: u32,
        ) -> Result<::quanta::Pulse, ::quanta::QuantaError> {
            // Allocate GPU fields for each Vec<T> in the struct
            #(#alloc_stmts)*

            // Upload data to GPU
            #(#upload_stmts)*

            // Create wave from compiled kernel binary
            let mut __wave = #wave_fn_name(device)?;

            // Bind buffer fields
            #(#bind_stmts)*

            // Set push constants (scalar fields)
            #(#push_stmts)*

            // Set const generic values
            #(#const_setters)*

            // Dispatch
            let mut __pulse = device.dispatch(&__wave, quarks)?;
            __pulse.wait()?;

            // Read back written fields
            #(#readback_stmts)*

            Ok(__pulse)
        }
    };

    expanded
}

/// Convert a scalar type name string to Rust type tokens.
fn scalar_type_to_rust_tokens(name: &str) -> TokenStream {
    match name {
        "f16" => quote! { u16 }, // f16 represented as u16 on CPU side
        "f32" => quote! { f32 },
        "f64" => quote! { f64 },
        "u8" => quote! { u8 },
        "u16" => quote! { u16 },
        "u32" => quote! { u32 },
        "u64" => quote! { u64 },
        "i8" => quote! { i8 },
        "i16" => quote! { i16 },
        "i32" => quote! { i32 },
        "i64" => quote! { i64 },
        "bool" => quote! { bool },
        _ => quote! { f32 }, // fallback
    }
}
