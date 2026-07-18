//! Shader parameter types, parsing, and body extraction.
//!
//! `pub` here is shared by the two DSL face crates; not a public API.
#![allow(dead_code)]

/// A parsed shader parameter — a vertex/fragment attribute, a `&T` uniform, or
/// a `&[T]` slice (storage-buffer array). `is_uniform` and `is_slice` are
/// mutually exclusive; `ty` on a slice is the element type.
pub struct ShaderParam {
    pub name: String,
    pub ty: ShaderType,
    pub is_uniform: bool,
    pub is_slice: bool,
}

/// How a parsed parameter binds: a plain value attribute, a `&T` uniform, or a
/// `&[T]` slice. Uniform and slice share one binding index space (see the
/// combined-cap check in `parse_shader_params`).
enum ParamClass {
    Value,
    Uniform,
    Slice,
}

/// Shader types understood by the vertex/fragment emitters.
#[derive(Clone, Copy)]
pub enum ShaderType {
    F32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
    Mat3,
    /// 32-bit unsigned integer scalar — an integer vertex attribute
    /// (`AttributeFormat::UInt`) or a flat-interpolated varying.
    U32,
}

impl ShaderType {
    pub fn msl_name(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::Vec2 => "float2",
            Self::Vec3 => "float3",
            Self::Vec4 => "float4",
            Self::Mat4 => "float4x4",
            Self::Mat3 => "float3x3",
            Self::U32 => "uint",
        }
    }

    pub fn wgsl_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Vec2 => "vec2<f32>",
            Self::Vec3 => "vec3<f32>",
            Self::Vec4 => "vec4<f32>",
            Self::Mat4 => "mat4x4<f32>",
            Self::Mat3 => "mat3x3<f32>",
            Self::U32 => "u32",
        }
    }
}

pub fn shader_type_from_ident(name: &str) -> Result<ShaderType, String> {
    match name {
        "f32" => Ok(ShaderType::F32),
        "u32" => Ok(ShaderType::U32),
        "Vec2" => Ok(ShaderType::Vec2),
        "Vec3" => Ok(ShaderType::Vec3),
        "Vec4" => Ok(ShaderType::Vec4),
        "Mat4" => Ok(ShaderType::Mat4),
        "Mat3" => Ok(ShaderType::Mat3),
        other => Err(format!("unsupported shader type: {}", other)),
    }
}

/// Parse function parameters into shader params plus texture params.
///
/// Value params (Vec2, Vec3, Vec4, f32) become attributes/inputs.
/// Reference params (&T) become uniform buffer bindings.
/// `&Texture2D` params become sampled textures: their slot is their
/// declaration order among texture params, and the macro rewrites
/// `sample(name, uv)` in the body to the slot form the emitters bind
/// (`[[texture(slot)]]`/`[[sampler(slot)]]` on Metal, descriptor
/// binding `slot + 8` on Vulkan).
pub fn parse_shader_params(
    func: &syn::ItemFn,
) -> Result<(Vec<ShaderParam>, Vec<String>), syn::Error> {
    let mut params: Vec<ShaderParam> = Vec::new();
    let mut textures = Vec::new();

    for arg in &func.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            let name = match pat_type.pat.as_ref() {
                syn::Pat::Ident(ident) => ident.ident.to_string(),
                _ => {
                    return Err(syn::Error::new_spanned(
                        &pat_type.pat,
                        "shader parameter must be a simple identifier",
                    ));
                }
            };

            if is_texture_type(&pat_type.ty) {
                match pat_type.ty.as_ref() {
                    syn::Type::Reference(_) => {}
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &pat_type.ty,
                            "texture parameters must be references: `&Texture2D`",
                        ));
                    }
                }
                if textures.len() >= 8 {
                    return Err(syn::Error::new_spanned(
                        &pat_type.pat,
                        "at most 8 texture parameters are supported (slots 0-7)",
                    ));
                }
                textures.push(name);
                continue;
            }

            let (ty, class) = parse_shader_type(&pat_type.ty)?;
            let (is_uniform, is_slice) = match class {
                ParamClass::Value => (false, false),
                ParamClass::Uniform => (true, false),
                ParamClass::Slice => (false, true),
            };
            // Uniforms and slices share ONE binding index space (bindings 0-7);
            // texture bindings start at 8, so more than 8 combined would collide.
            if (is_uniform || is_slice)
                && params.iter().filter(|p| p.is_uniform || p.is_slice).count() >= 8
            {
                return Err(syn::Error::new_spanned(
                    &pat_type.pat,
                    "at most 8 combined uniform and slice parameters are supported \
                     (bindings 0-7; texture bindings start at 8)",
                ));
            }
            params.push(ShaderParam {
                name,
                ty,
                is_uniform,
                is_slice,
            });
        }
    }

    Ok((params, textures))
}

/// Whether the (possibly referenced) type is the `Texture2D` marker.
fn is_texture_type(ty: &syn::Type) -> bool {
    let inner = match ty {
        syn::Type::Reference(r) => r.elem.as_ref(),
        other => other,
    };
    if let syn::Type::Path(path) = inner
        && let Some(seg) = path.path.segments.last()
    {
        return seg.ident == "Texture2D";
    }
    false
}

/// Rewrite `sample(param_name, ...)` calls to the canonical slot form
/// `sample(N, ...)` the emitters recognize. The body arrives as a
/// proc-macro token string, so spacing around `(` and `,` varies —
/// all four combinations are normalized to the compact form.
pub fn rewrite_texture_names(body: &str, textures: &[String]) -> String {
    let mut s = body.to_string();
    for (slot, name) in textures.iter().enumerate() {
        let canonical = format!("sample({slot},");
        for pat in [
            format!("sample ({name} ,"),
            format!("sample({name} ,"),
            format!("sample ({name},"),
            format!("sample({name},"),
        ] {
            s = s.replace(&pat, &canonical);
        }
    }
    s
}

/// Parse a type into (element `ShaderType`, `ParamClass`).
/// `&[T]` → slice, `&T` → uniform, `T` → attribute/input.
fn parse_shader_type(ty: &syn::Type) -> Result<(ShaderType, ParamClass), syn::Error> {
    match ty {
        syn::Type::Reference(ref_ty) => match ref_ty.elem.as_ref() {
            // `&[T]` — a storage-buffer array. The element type is restricted to
            // f32/Vec2/Vec4 (the runtime binds these as tightly-packed SSBOs);
            // anything else (Vec3/Mat4/nested/`u32`) is a compile error.
            syn::Type::Slice(slice_ty) => {
                let elem = parse_slice_element(&slice_ty.elem)?;
                Ok((elem, ParamClass::Slice))
            }
            other => {
                let inner = parse_shader_type_inner(other)?;
                Ok((inner, ParamClass::Uniform))
            }
        },
        _ => {
            let inner = parse_shader_type_inner(ty)?;
            Ok((inner, ParamClass::Value))
        }
    }
}

/// Parse a slice element type. Only `f32`, `Vec2`, and `Vec4` are allowed as
/// `&[T]` shader-parameter element types.
fn parse_slice_element(ty: &syn::Type) -> Result<ShaderType, syn::Error> {
    let inner = parse_shader_type_inner(ty)?;
    match inner {
        ShaderType::F32 | ShaderType::Vec2 | ShaderType::Vec4 => Ok(inner),
        _ => Err(syn::Error::new_spanned(
            ty,
            "slice parameters support only `&[f32]`, `&[Vec2]`, and `&[Vec4]` \
             element types",
        )),
    }
}

fn parse_shader_type_inner(ty: &syn::Type) -> Result<ShaderType, syn::Error> {
    match ty {
        syn::Type::Path(path) => {
            let ident = path
                .path
                .segments
                .last()
                .ok_or_else(|| syn::Error::new_spanned(path, "empty type path"))?;
            shader_type_from_ident(&ident.ident.to_string())
                .map_err(|msg| syn::Error::new_spanned(&ident.ident, msg))
        }
        _ => Err(syn::Error::new_spanned(ty, "unsupported shader type")),
    }
}

/// Parse the return type of a shader function.
pub fn parse_return_type(func: &syn::ItemFn) -> Result<ShaderType, syn::Error> {
    match &func.sig.output {
        syn::ReturnType::Type(_, ty) => parse_shader_type_inner(ty),
        syn::ReturnType::Default => Err(syn::Error::new_spanned(
            &func.sig.ident,
            "shader must have a return type",
        )),
    }
}

/// Extract the function body as a source string for text-based translation.
pub fn extract_body_source(func: &syn::ItemFn) -> String {
    use quote::ToTokens;
    let mut body = String::new();
    for stmt in &func.block.stmts {
        body.push_str(&stmt.to_token_stream().to_string());
        body.push('\n');
    }
    body
}
