//! Implementation of `#[quanta::gpu_type]` — marks a struct as GPU-compatible
//! and generates metadata for kernel field access, MSL, and WGSL declarations.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Fields, ItemStruct, Type};

/// Information about a single field in a GPU struct.
struct GpuField {
    name: String,
    type_str: String,
    msl_type: String,
    wgsl_type: String,
}

use crate::crate_path::CratePath;

/// Generate all output tokens for a `#[quanta::gpu_type]` struct.
pub(crate) fn expand_gpu_type(
    input: &ItemStruct,
    cp: &CratePath,
) -> Result<TokenStream, syn::Error> {
    let fields = match &input.fields {
        Fields::Named(named) => &named.named,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "#[quanta::gpu_type] only supports structs with named fields",
            ));
        }
    };

    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();
    let struct_name_upper = struct_name_str.to_uppercase();

    // Parse all fields
    let mut gpu_fields = Vec::new();
    let mut field_idents = Vec::new();
    for field in fields {
        let ident = field
            .ident
            .as_ref()
            .ok_or_else(|| syn::Error::new_spanned(field, "unnamed field"))?;
        let gpu_field = parse_gpu_field(&ident.to_string(), &field.ty)?;
        gpu_fields.push(gpu_field);
        field_idents.push(ident.clone());
    }

    // Build GPU_FIELDS entries: (name, type_str, byte_offset). The offset is
    // the compiler's, not ours: `offset_of!` on the emitted #[repr(C)] struct
    // is exact for every field type — nested gpu_type structs included, whose
    // sizes a token-level macro cannot know.
    let field_entries: Vec<TokenStream> = gpu_fields
        .iter()
        .zip(field_idents.iter())
        .map(|(f, ident)| {
            let name = &f.name;
            let ty = &f.type_str;
            quote! { (#name, #ty, core::mem::offset_of!(Self, #ident)) }
        })
        .collect();

    // Build MSL struct declaration
    let mut msl_lines = Vec::new();
    msl_lines.push(format!("struct {} {{", struct_name_str));
    for f in &gpu_fields {
        msl_lines.push(format!("    {} {};", f.msl_type, f.name));
    }
    msl_lines.push("};".to_string());
    let msl_decl = msl_lines.join("\n") + "\n";

    // Build WGSL struct declaration
    let mut wgsl_lines = Vec::new();
    wgsl_lines.push(format!("struct {} {{", struct_name_str));
    for f in gpu_fields.iter() {
        let comma = ",";
        wgsl_lines.push(format!("    {}: {}{}", f.name, f.wgsl_type, comma));
    }
    wgsl_lines.push("};".to_string());
    let wgsl_decl = wgsl_lines.join("\n") + "\n";

    let msl_const_name = format_ident!("__QUANTA_GPU_TYPE_{}", struct_name_upper);
    let wgsl_const_name = format_ident!("__QUANTA_GPU_TYPE_{}_WGSL", struct_name_upper);

    // Re-emit the struct with #[repr(C)] and #[derive(Copy, Clone)]
    // Check if they are already present
    let has_repr_c = input.attrs.iter().any(|a| {
        if a.path().is_ident("repr") {
            a.parse_args::<syn::Ident>()
                .map(|i| i == "C")
                .unwrap_or(false)
        } else {
            false
        }
    });

    let has_derive_copy = input.attrs.iter().any(|a| {
        if a.path().is_ident("derive") {
            a.to_token_stream().to_string().contains("Copy")
        } else {
            false
        }
    });

    // Collect existing attributes, filtering out the ones we will add
    let existing_attrs: Vec<_> = input
        .attrs
        .iter()
        .filter(|a| !a.path().is_ident("gpu_type"))
        .collect();

    let repr_attr = if has_repr_c {
        quote! {}
    } else {
        quote! { #[repr(C)] }
    };

    let derive_attr = if has_derive_copy {
        quote! {}
    } else {
        quote! { #[derive(Copy, Clone)] }
    };

    let vis = &input.vis;
    let field_defs = fields.iter().map(|f| {
        let attrs = &f.attrs;
        let vis = &f.vis;
        let ident = &f.ident;
        let ty = &f.ty;
        quote! { #(#attrs)* #vis #ident: #ty }
    });

    let generics = &input.generics;
    let krate = cp.types();

    let msl_const_doc = format!(
        " MSL declaration of [`{struct_name_str}`], for hand-written Metal \
         shader sources that name the type."
    );
    let wgsl_const_doc = format!(
        " WGSL declaration of [`{struct_name_str}`], for hand-written WGSL \
         shader sources that name the type."
    );

    Ok(quote! {
        #(#existing_attrs)*
        #repr_attr
        #derive_attr
        #vis struct #struct_name #generics {
            #(#field_defs,)*
        }

        impl #struct_name {
            /// Byte size of this struct on the GPU (host `repr(C)` layout).
            pub const GPU_SIZE: usize = core::mem::size_of::<Self>();
            /// Field metadata: `(name, type_string, byte_offset)` per field,
            /// offsets resolved by the compiler from the `repr(C)` layout.
            pub const GPU_FIELDS: &'static [(&'static str, &'static str, usize)] = &[
                #(#field_entries,)*
            ];
        }

        impl #krate::GpuType for #struct_name {
            fn gpu_size() -> usize { core::mem::size_of::<Self>() }
            fn scalar_type() -> #krate::ScalarType { #krate::ScalarType::U8 }
        }

        #[doc = #msl_const_doc]
        pub const #msl_const_name: &str = #msl_decl;
        #[doc = #wgsl_const_doc]
        pub const #wgsl_const_name: &str = #wgsl_decl;
    })
}

use syn::Attribute;

/// Check whether a `#[derive(...)]` attribute contains a specific derive name.
trait AttrExt {
    fn to_token_stream(&self) -> TokenStream;
}

impl AttrExt for Attribute {
    fn to_token_stream(&self) -> TokenStream {
        quote! { #self }
    }
}

/// Parse a single field type into GPU metadata.
fn parse_gpu_field(name: &str, ty: &Type) -> Result<GpuField, syn::Error> {
    match ty {
        // Array types: [T; N]
        Type::Array(arr) => {
            let elem_type = type_to_scalar_str(&arr.elem)?;
            let len = parse_array_len(&arr.len)?;
            let type_str = format!("[{}; {}]", elem_type.rust_name, len);

            let (msl_type, wgsl_type) = array_gpu_type(&elem_type, len, name)?;

            Ok(GpuField {
                name: name.to_string(),
                type_str,
                msl_type,
                wgsl_type,
            })
        }
        // Scalar or named struct types
        Type::Path(_) => {
            let info = type_to_scalar_str(ty)?;
            Ok(GpuField {
                name: name.to_string(),
                type_str: info.rust_name.clone(),
                msl_type: info.msl_name.clone(),
                wgsl_type: info.wgsl_name.clone(),
            })
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            format!("unsupported GPU field type for '{}'", name),
        )),
    }
}

/// Info about a scalar/named type.
struct TypeInfo {
    rust_name: String,
    msl_name: String,
    wgsl_name: String,
}

/// Map a Rust type to its GPU type info.
fn type_to_scalar_str(ty: &Type) -> Result<TypeInfo, syn::Error> {
    match ty {
        Type::Path(path) => {
            let ident = path
                .path
                .segments
                .last()
                .ok_or_else(|| syn::Error::new_spanned(path, "empty type path"))?;
            let name = ident.ident.to_string();
            match name.as_str() {
                "f32" => Ok(TypeInfo {
                    rust_name: "f32".into(),
                    msl_name: "float".into(),
                    wgsl_name: "f32".into(),
                }),
                "f64" => Ok(TypeInfo {
                    rust_name: "f64".into(),
                    msl_name: "double".into(),
                    wgsl_name: "f64".into(),
                }),
                "u32" => Ok(TypeInfo {
                    rust_name: "u32".into(),
                    msl_name: "uint".into(),
                    wgsl_name: "u32".into(),
                }),
                "i32" => Ok(TypeInfo {
                    rust_name: "i32".into(),
                    msl_name: "int".into(),
                    wgsl_name: "i32".into(),
                }),
                "u8" => Ok(TypeInfo {
                    rust_name: "u8".into(),
                    msl_name: "uint8_t".into(),
                    wgsl_name: "u32".into(),
                }),
                "bool" => Ok(TypeInfo {
                    rust_name: "bool".into(),
                    msl_name: "bool".into(),
                    wgsl_name: "bool".into(),
                }),
                "u64" => Ok(TypeInfo {
                    rust_name: "u64".into(),
                    msl_name: "ulong".into(),
                    wgsl_name: "u32".into(), // WGSL has limited u64 support
                }),
                "i64" => Ok(TypeInfo {
                    rust_name: "i64".into(),
                    msl_name: "long".into(),
                    wgsl_name: "i32".into(), // WGSL has limited i64 support
                }),
                "u16" => Ok(TypeInfo {
                    rust_name: "u16".into(),
                    msl_name: "ushort".into(),
                    wgsl_name: "u32".into(),
                }),
                "i16" => Ok(TypeInfo {
                    rust_name: "i16".into(),
                    msl_name: "short".into(),
                    wgsl_name: "i32".into(),
                }),
                // Nested struct: referenced by name in the shader
                // declarations; its size and alignment never enter the
                // macro — GPU_FIELDS offsets come from `offset_of!` and
                // GPU_SIZE from `size_of`, so the compiler owns the layout.
                other => Ok(TypeInfo {
                    rust_name: other.into(),
                    msl_name: other.into(),
                    wgsl_name: other.into(),
                }),
            }
        }
        _ => Err(syn::Error::new_spanned(ty, "unsupported GPU type")),
    }
}

/// Map an array [T; N] to its MSL/WGSL spelling (vectorized where
/// applicable). Naming only: sizes and offsets come from the compiler
/// via `size_of`/`offset_of!` on the emitted #[repr(C)] struct.
fn array_gpu_type(
    elem: &TypeInfo,
    len: usize,
    _field_name: &str,
) -> Result<(String, String), syn::Error> {
    let elem_rust = elem.rust_name.as_str();

    // Check for vector/matrix special cases (MSL/WGSL naming only)
    match (elem_rust, len) {
        // float vectors
        ("f32", 2) => Ok(("float2".into(), "vec2<f32>".into())),
        ("f32", 3) => Ok(("float3".into(), "vec3<f32>".into())),
        ("f32", 4) => Ok(("float4".into(), "vec4<f32>".into())),
        // float matrices
        ("f32", 9) => Ok(("float3x3".into(), "mat3x3<f32>".into())),
        ("f32", 16) => Ok(("float4x4".into(), "mat4x4<f32>".into())),

        // uint vectors
        ("u32", 2) => Ok(("uint2".into(), "vec2<u32>".into())),
        ("u32", 3) => Ok(("uint3".into(), "vec3<u32>".into())),
        ("u32", 4) => Ok(("uint4".into(), "vec4<u32>".into())),

        // int vectors
        ("i32", 2) => Ok(("int2".into(), "vec2<i32>".into())),
        ("i32", 3) => Ok(("int3".into(), "vec3<i32>".into())),
        ("i32", 4) => Ok(("int4".into(), "vec4<i32>".into())),

        // f64 vectors
        ("f64", 2) => Ok(("double2".into(), "vec2<f64>".into())),
        ("f64", 3) => Ok(("double3".into(), "vec3<f64>".into())),
        ("f64", 4) => Ok(("double4".into(), "vec4<f64>".into())),

        // Generic arrays — not vectorized
        _ => {
            let msl = format!("{} [{}]", elem.msl_name, len);
            let wgsl = format!("array<{}, {}>", elem.wgsl_name, len);
            Ok((msl, wgsl))
        }
    }
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
            "array length must be a literal (const generics not supported in GPU types)",
        )),
    }
}
