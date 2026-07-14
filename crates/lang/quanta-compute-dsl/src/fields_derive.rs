//! Implementation of `#[derive(Fields)]` — generates GPU-side metadata for
//! kernel data structs. Classifies each field as either a Field (Vec<T>) or
//! a push constant (scalar), producing slot metadata that the `#[quanta::kernel]`
//! macro uses to generate upload/bind/dispatch code.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Fields, ItemStruct, Type};

/// Classification of a struct field for GPU dispatch.
enum FieldKind {
    /// `Vec<T>` — will become a GPU `Field<T>` (storage buffer).
    GpuField { elem_type: String },
    /// Scalar value — will become a push constant.
    PushConstant { type_str: String },
}

/// Generate the `Fields` derive implementation for a struct.
///
/// Produces metadata constants and functions:
/// - `FIELD_COUNT: usize` — number of `Vec<T>` fields (GPU storage buffers)
/// - `PUSH_CONSTANT_COUNT: usize` — number of scalar fields (push constants)
/// - `fn field_names() -> &'static [&'static str]` — names of Vec fields
/// - `fn field_types() -> &'static [&'static str]` — element types of Vec fields ("f32", etc.)
/// - `fn push_constant_names() -> &'static [&'static str]` — names of scalar fields
/// - `fn push_constant_types() -> &'static [&'static str]` — types of scalar fields
pub(crate) fn expand_fields_derive(input: &ItemStruct) -> Result<TokenStream, syn::Error> {
    let fields = match &input.fields {
        Fields::Named(named) => &named.named,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "#[derive(Fields)] only supports structs with named fields",
            ));
        }
    };

    let struct_name = &input.ident;

    let mut gpu_field_names = Vec::new();
    let mut gpu_field_types = Vec::new();
    let mut push_names = Vec::new();
    let mut push_types = Vec::new();

    for field in fields {
        let field_name = field
            .ident
            .as_ref()
            .ok_or_else(|| syn::Error::new_spanned(field, "unnamed field"))?
            .to_string();

        match classify_field(&field.ty)? {
            FieldKind::GpuField { elem_type } => {
                gpu_field_names.push(field_name);
                gpu_field_types.push(elem_type);
            }
            FieldKind::PushConstant { type_str } => {
                push_names.push(field_name);
                push_types.push(type_str);
            }
        }
    }

    let field_count = gpu_field_names.len();
    let push_count = push_names.len();

    Ok(quote! {
        impl #struct_name {
            /// Number of `Vec<T>` fields — each becomes a GPU storage buffer (Field).
            pub const FIELD_COUNT: usize = #field_count;

            /// Number of scalar fields — each becomes a push constant.
            pub const PUSH_CONSTANT_COUNT: usize = #push_count;

            /// Names of the Vec<T> fields (GPU storage buffers).
            pub fn field_names() -> &'static [&'static str] {
                &[#(#gpu_field_names),*]
            }

            /// Element type names of the Vec<T> fields (e.g., "f32", "u32").
            pub fn field_types() -> &'static [&'static str] {
                &[#(#gpu_field_types),*]
            }

            /// Names of the scalar fields (push constants).
            pub fn push_constant_names() -> &'static [&'static str] {
                &[#(#push_names),*]
            }

            /// Type names of the scalar fields (e.g., "u32", "f32").
            pub fn push_constant_types() -> &'static [&'static str] {
                &[#(#push_types),*]
            }
        }
    })
}

/// Classify a field type as either a GPU field (`Vec<T>`) or a push constant (scalar).
fn classify_field(ty: &Type) -> Result<FieldKind, syn::Error> {
    match ty {
        Type::Path(path) => {
            let segment = path
                .path
                .segments
                .last()
                .ok_or_else(|| syn::Error::new_spanned(path, "empty type path"))?;

            let name = segment.ident.to_string();

            // Check for Vec<T>
            if name == "Vec" {
                let elem_type = extract_vec_element(segment)?;
                return Ok(FieldKind::GpuField { elem_type });
            }

            // Scalars
            if is_gpu_scalar(&name) {
                return Ok(FieldKind::PushConstant { type_str: name });
            }

            Err(syn::Error::new_spanned(
                ty,
                format!(
                    "unsupported Fields type `{}`. \
                     Supported: Vec<T> (GPU field), f32/u32/i32/u64/i64/f64/u8/u16/i16/bool (push constant)",
                    name
                ),
            ))
        }
        Type::Array(arr) => {
            // Fixed-size arrays are push constants (e.g., [f32; 4] for a vec4 uniform)
            let elem_type = match arr.elem.as_ref() {
                Type::Path(p) => p
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .ok_or_else(|| syn::Error::new_spanned(p, "empty array element type"))?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        &arr.elem,
                        "unsupported array element type",
                    ));
                }
            };

            let len = parse_array_len(&arr.len)?;
            let type_str = format!("[{}; {}]", elem_type, len);
            Ok(FieldKind::PushConstant { type_str })
        }
        _ => Err(syn::Error::new_spanned(ty, "unsupported Fields type")),
    }
}

/// Extract the element type name from `Vec<T>`.
fn extract_vec_element(segment: &syn::PathSegment) -> Result<String, syn::Error> {
    match &segment.arguments {
        syn::PathArguments::AngleBracketed(args) => {
            if let Some(syn::GenericArgument::Type(Type::Path(p))) = args.args.first()
                && let Some(seg) = p.path.segments.last()
            {
                return Ok(seg.ident.to_string());
            }
            Err(syn::Error::new_spanned(
                &segment.arguments,
                "Vec must have a single type argument (e.g., Vec<f32>)",
            ))
        }
        _ => Err(syn::Error::new_spanned(
            segment,
            "Vec must have angle-bracketed type argument",
        )),
    }
}

/// Check if a type name is a GPU-compatible scalar.
fn is_gpu_scalar(name: &str) -> bool {
    matches!(
        name,
        "f32" | "f64" | "u32" | "i32" | "u8" | "u16" | "i16" | "u64" | "i64" | "bool" | "usize"
    )
}

/// Parse the length expression of an array type.
fn parse_array_len(expr: &syn::Expr) -> Result<usize, syn::Error> {
    match expr {
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(int) => int
                .base10_parse::<usize>()
                .map_err(|e| syn::Error::new_spanned(int, e)),
            _ => Err(syn::Error::new_spanned(
                expr,
                "array length must be an integer literal",
            )),
        },
        _ => Err(syn::Error::new_spanned(
            expr,
            "array length must be a literal",
        )),
    }
}
