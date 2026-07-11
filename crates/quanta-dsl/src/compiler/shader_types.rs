//! Shader parameter types, parsing, and body extraction.
#![allow(dead_code)]

/// A parsed shader parameter — either a vertex/fragment attribute or a uniform.
pub(crate) struct ShaderParam {
    pub(crate) name: String,
    pub(crate) ty: ShaderType,
    pub(crate) is_uniform: bool,
}

/// Shader types understood by the vertex/fragment emitters.
#[derive(Clone, Copy)]
pub(crate) enum ShaderType {
    F32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
    Mat3,
}

impl ShaderType {
    pub(crate) fn msl_name(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::Vec2 => "float2",
            Self::Vec3 => "float3",
            Self::Vec4 => "float4",
            Self::Mat4 => "float4x4",
            Self::Mat3 => "float3x3",
        }
    }

    pub(crate) fn wgsl_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Vec2 => "vec2<f32>",
            Self::Vec3 => "vec3<f32>",
            Self::Vec4 => "vec4<f32>",
            Self::Mat4 => "mat4x4<f32>",
            Self::Mat3 => "mat3x3<f32>",
        }
    }
}

pub(crate) fn shader_type_from_ident(name: &str) -> Result<ShaderType, String> {
    match name {
        "f32" => Ok(ShaderType::F32),
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
pub(crate) fn parse_shader_params(
    func: &syn::ItemFn,
) -> Result<(Vec<ShaderParam>, Vec<String>), syn::Error> {
    let mut params = Vec::new();
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

            let (ty, is_uniform) = parse_shader_type(&pat_type.ty)?;
            params.push(ShaderParam {
                name,
                ty,
                is_uniform,
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
pub(crate) fn rewrite_texture_names(body: &str, textures: &[String]) -> String {
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

/// Parse a type into (ShaderType, is_uniform).
/// `&T` → uniform, `T` → attribute/input.
fn parse_shader_type(ty: &syn::Type) -> Result<(ShaderType, bool), syn::Error> {
    match ty {
        syn::Type::Reference(ref_ty) => {
            let inner = parse_shader_type_inner(&ref_ty.elem)?;
            Ok((inner, true))
        }
        _ => {
            let inner = parse_shader_type_inner(ty)?;
            Ok((inner, false))
        }
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
pub(crate) fn parse_return_type(func: &syn::ItemFn) -> Result<ShaderType, syn::Error> {
    match &func.sig.output {
        syn::ReturnType::Type(_, ty) => parse_shader_type_inner(ty),
        syn::ReturnType::Default => Err(syn::Error::new_spanned(
            &func.sig.ident,
            "shader must have a return type",
        )),
    }
}

/// Extract the function body as a source string for text-based translation.
pub(crate) fn extract_body_source(func: &syn::ItemFn) -> String {
    use quote::ToTokens;
    let mut body = String::new();
    for stmt in &func.block.stmts {
        body.push_str(&stmt.to_token_stream().to_string());
        body.push('\n');
    }
    body
}
