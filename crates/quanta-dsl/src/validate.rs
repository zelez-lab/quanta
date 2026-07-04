//! Validate that a Rust function is GPU-safe.

use syn::{FnArg, ItemFn, ReturnType, Type};

/// Validate a function for GPU kernel constraints.
pub fn validate_kernel(func: &ItemFn) -> Result<(), syn::Error> {
    // Must return nothing — kernels write to output fields
    if !matches!(func.sig.output, ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &func.sig.output,
            "GPU kernel must return () — write results to &mut fields instead",
        ));
    }

    // No async
    if func.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            func.sig.fn_token,
            "GPU kernel cannot be async",
        ));
    }

    // No unsafe
    if func.sig.unsafety.is_some() {
        return Err(syn::Error::new_spanned(
            func.sig.fn_token,
            "GPU kernel cannot be unsafe",
        ));
    }

    // Allow const generics, reject type/lifetime generics
    for param in &func.sig.generics.params {
        if !matches!(param, syn::GenericParam::Const(_)) {
            return Err(syn::Error::new_spanned(
                param,
                "GPU kernel only supports const generic parameters (not type or lifetime)",
            ));
        }
    }

    // Validate parameters
    for arg in &func.sig.inputs {
        match arg {
            FnArg::Receiver(_) => {
                return Err(syn::Error::new_spanned(
                    arg,
                    "GPU kernel cannot have self parameter",
                ));
            }
            FnArg::Typed(pat_type) => {
                validate_param_type(&pat_type.ty)?;
            }
        }
    }

    Ok(())
}

fn validate_param_type(ty: &Type) -> Result<(), syn::Error> {
    match ty {
        // &[T] or &mut [T] — field references
        Type::Reference(_) => Ok(()),
        // Scalar types — push constants
        Type::Path(_) => Ok(()),
        _ => Err(syn::Error::new_spanned(
            ty,
            "GPU kernel parameter must be &[T] (read field), &mut [T] (write field), or a scalar type (push constant)",
        )),
    }
}
