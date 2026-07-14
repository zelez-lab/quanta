//! Implementation of `#[derive(Uniforms)]` — marks a struct as GPU-compatible
//! for use as a uniform buffer. Generates `#[repr(C)]` enforcement, `GpuType` impl,
//! and byte-level metadata (size, field offsets).
//!
//! Reuses GPU type mapping logic from `gpu_type.rs`.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Fields, ItemStruct, Type};

/// Generate the `Uniforms` derive implementation for a struct.
///
/// Produces:
/// - `GPU_SIZE: usize` — byte size of the struct
/// - `GPU_FIELDS: &[(&str, &str, usize)]` — (name, type_str, byte_offset) for each field
/// - `impl GpuType` — marks the struct as GPU-compatible
/// - MSL/WGSL struct declaration constants
pub(crate) fn expand_uniforms_derive(input: &ItemStruct) -> Result<TokenStream, syn::Error> {
    let cp = crate::crate_path::crate_from_container_attrs(&input.attrs);
    let krate = cp.types();

    let fields = match &input.fields {
        Fields::Named(named) => &named.named,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "#[derive(Uniforms)] only supports structs with named fields",
            ));
        }
    };

    // Verify #[repr(C)] is present
    let has_repr_c = input.attrs.iter().any(|a| {
        if a.path().is_ident("repr") {
            a.parse_args::<syn::Ident>()
                .map(|i| i == "C")
                .unwrap_or(false)
        } else {
            false
        }
    });

    if !has_repr_c {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(Uniforms)] requires #[repr(C)] on the struct",
        ));
    }

    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();
    let struct_name_upper = struct_name_str.to_uppercase();

    // Parse all fields and compute offsets (repr(C) layout)
    let mut field_infos = Vec::new();
    let mut current_offset: usize = 0;
    let mut max_align: usize = 1;

    for field in fields {
        let field_name = field
            .ident
            .as_ref()
            .ok_or_else(|| syn::Error::new_spanned(field, "unnamed field"))?
            .to_string();

        let info = parse_uniform_field(&field_name, &field.ty)?;

        // Align to field alignment
        let misalign = current_offset % info.align;
        if misalign != 0 {
            current_offset += info.align - misalign;
        }

        field_infos.push((
            field_name,
            info.type_str,
            info.msl_type,
            info.wgsl_type,
            current_offset,
        ));
        current_offset += info.size;
        if info.align > max_align {
            max_align = info.align;
        }
    }

    // Pad to struct alignment (compute final size for validation, not used in output
    // since we rely on core::mem::size_of::<Self>() for the canonical value).
    let _misalign = current_offset % max_align;

    // Build GPU_FIELDS entries
    let field_entries: Vec<TokenStream> = field_infos
        .iter()
        .map(|(name, ty, _, _, off)| {
            quote! { (#name, #ty, #off) }
        })
        .collect();

    // Build MSL struct declaration
    let mut msl_lines = Vec::new();
    msl_lines.push(format!("struct {} {{", struct_name_str));
    for (name, _, msl_type, _, _) in &field_infos {
        msl_lines.push(format!("    {} {};", msl_type, name));
    }
    msl_lines.push("};".to_string());
    let msl_decl = msl_lines.join("\n") + "\n";

    // Build WGSL struct declaration
    let mut wgsl_lines = Vec::new();
    wgsl_lines.push(format!("struct {} {{", struct_name_str));
    for (name, _, _, wgsl_type, _) in &field_infos {
        wgsl_lines.push(format!("    {}: {},", name, wgsl_type));
    }
    wgsl_lines.push("};".to_string());
    let wgsl_decl = wgsl_lines.join("\n") + "\n";

    let msl_const_name = format_ident!("__QUANTA_UNIFORMS_{}", struct_name_upper);
    let wgsl_const_name = format_ident!("__QUANTA_UNIFORMS_{}_WGSL", struct_name_upper);

    Ok(quote! {
        impl #struct_name {
            /// Byte size of this uniform struct (matches GPU layout).
            pub const GPU_SIZE: usize = core::mem::size_of::<Self>();

            /// Field metadata: (name, type_string, byte_offset) for each field.
            pub const GPU_FIELDS: &'static [(&'static str, &'static str, usize)] = &[
                #(#field_entries,)*
            ];
        }

        impl #krate::GpuType for #struct_name {
            fn gpu_size() -> usize { core::mem::size_of::<Self>() }
            fn scalar_type() -> #krate::ScalarType { #krate::ScalarType::U8 }
        }

        #[doc(hidden)]
        pub const #msl_const_name: &str = #msl_decl;
        #[doc(hidden)]
        pub const #wgsl_const_name: &str = #wgsl_decl;
    })
}

/// Metadata for a single uniform field.
struct UniformFieldInfo {
    type_str: String,
    msl_type: String,
    wgsl_type: String,
    size: usize,
    align: usize,
}

/// Parse a field type into uniform metadata, supporting scalars and arrays.
fn parse_uniform_field(name: &str, ty: &Type) -> Result<UniformFieldInfo, syn::Error> {
    match ty {
        Type::Array(arr) => {
            let elem = scalar_info(&arr.elem)?;
            let len = parse_array_len(&arr.len)?;
            let type_str = format!("[{}; {}]", elem.type_str, len);
            let (msl_type, wgsl_type, size) = array_gpu_types(&elem, len);

            Ok(UniformFieldInfo {
                type_str,
                msl_type,
                wgsl_type,
                size,
                align: elem.align,
            })
        }
        Type::Path(_) => {
            let info = scalar_info(ty)?;
            Ok(UniformFieldInfo {
                type_str: info.type_str.clone(),
                msl_type: info.msl_type.clone(),
                wgsl_type: info.wgsl_type.clone(),
                size: info.size,
                align: info.align,
            })
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            format!("unsupported uniform field type for '{}'", name),
        )),
    }
}

/// Info about a scalar type.
struct ScalarInfo {
    type_str: String,
    msl_type: String,
    wgsl_type: String,
    size: usize,
    align: usize,
}

/// Map a Rust scalar type to GPU type info.
fn scalar_info(ty: &Type) -> Result<ScalarInfo, syn::Error> {
    let Type::Path(path) = ty else {
        return Err(syn::Error::new_spanned(ty, "expected a path type"));
    };
    let ident = path
        .path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new_spanned(path, "empty type path"))?;
    let name = ident.ident.to_string();

    match name.as_str() {
        "f32" => Ok(ScalarInfo {
            type_str: "f32".into(),
            msl_type: "float".into(),
            wgsl_type: "f32".into(),
            size: 4,
            align: 4,
        }),
        "f64" => Ok(ScalarInfo {
            type_str: "f64".into(),
            msl_type: "double".into(),
            wgsl_type: "f64".into(),
            size: 8,
            align: 8,
        }),
        "u32" => Ok(ScalarInfo {
            type_str: "u32".into(),
            msl_type: "uint".into(),
            wgsl_type: "u32".into(),
            size: 4,
            align: 4,
        }),
        "i32" => Ok(ScalarInfo {
            type_str: "i32".into(),
            msl_type: "int".into(),
            wgsl_type: "i32".into(),
            size: 4,
            align: 4,
        }),
        "u8" => Ok(ScalarInfo {
            type_str: "u8".into(),
            msl_type: "uint8_t".into(),
            wgsl_type: "u32".into(),
            size: 1,
            align: 1,
        }),
        "bool" => Ok(ScalarInfo {
            type_str: "bool".into(),
            msl_type: "bool".into(),
            wgsl_type: "bool".into(),
            size: 1,
            align: 1,
        }),
        "u64" => Ok(ScalarInfo {
            type_str: "u64".into(),
            msl_type: "ulong".into(),
            wgsl_type: "u32".into(),
            size: 8,
            align: 8,
        }),
        "i64" => Ok(ScalarInfo {
            type_str: "i64".into(),
            msl_type: "long".into(),
            wgsl_type: "i32".into(),
            size: 8,
            align: 8,
        }),
        "u16" => Ok(ScalarInfo {
            type_str: "u16".into(),
            msl_type: "ushort".into(),
            wgsl_type: "u32".into(),
            size: 2,
            align: 2,
        }),
        "i16" => Ok(ScalarInfo {
            type_str: "i16".into(),
            msl_type: "short".into(),
            wgsl_type: "i32".into(),
            size: 2,
            align: 2,
        }),
        other => Err(syn::Error::new_spanned(
            ty,
            format!("unsupported uniform type `{}`", other),
        )),
    }
}

/// Map an array [T; N] to GPU type strings and byte size.
fn array_gpu_types(elem: &ScalarInfo, len: usize) -> (String, String, usize) {
    let elem_rust = elem.type_str.as_str();
    match (elem_rust, len) {
        ("f32", 2) => ("float2".into(), "vec2<f32>".into(), 8),
        ("f32", 3) => ("float3".into(), "vec3<f32>".into(), 12),
        ("f32", 4) => ("float4".into(), "vec4<f32>".into(), 16),
        ("f32", 9) => ("float3x3".into(), "mat3x3<f32>".into(), 36),
        ("f32", 16) => ("float4x4".into(), "mat4x4<f32>".into(), 64),
        ("u32", 2) => ("uint2".into(), "vec2<u32>".into(), 8),
        ("u32", 3) => ("uint3".into(), "vec3<u32>".into(), 12),
        ("u32", 4) => ("uint4".into(), "vec4<u32>".into(), 16),
        ("i32", 2) => ("int2".into(), "vec2<i32>".into(), 8),
        ("i32", 3) => ("int3".into(), "vec3<i32>".into(), 12),
        ("i32", 4) => ("int4".into(), "vec4<i32>".into(), 16),
        _ => {
            let total_size = elem.size * len;
            let msl = format!("{} [{}]", elem.msl_type, len);
            let wgsl = format!("array<{}, {}>", elem.wgsl_type, len);
            (msl, wgsl, total_size)
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
            "array length must be a literal",
        )),
    }
}
