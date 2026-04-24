//! Vertex/fragment shader MSL and WGSL text emitters.

use super::shader_types::{ShaderParam, ShaderType};

/// Translate a Rust shader body to MSL.
fn translate_body_to_msl_shader(src: &str, _params: &[ShaderParam]) -> String {
    let mut s = src.to_string();
    s = s.replace("Vec4 :: new", "float4");
    s = s.replace("Vec4::new", "float4");
    s = s.replace("Vec3 :: new", "float3");
    s = s.replace("Vec3::new", "float3");
    s = s.replace("Vec2 :: new", "float2");
    s = s.replace("Vec2::new", "float2");
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s = s.replace(" as i32", "");
    s = s.replace("let mut ", "auto ");
    s = s.replace("let ", "auto ");
    s
}

/// Translate a Rust shader body to WGSL.
fn translate_body_to_wgsl_shader(src: &str, params: &[ShaderParam], _is_vertex: bool) -> String {
    let mut s = src.to_string();
    s = s.replace("Vec4 :: new", "vec4<f32>");
    s = s.replace("Vec4::new", "vec4<f32>");
    s = s.replace("Vec3 :: new", "vec3<f32>");
    s = s.replace("Vec3::new", "vec3<f32>");
    s = s.replace("Vec2 :: new", "vec2<f32>");
    s = s.replace("Vec2::new", "vec2<f32>");
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s = s.replace("let mut ", "var ");

    for p in params {
        if !p.is_uniform {
            let from = &p.name;
            let to = format!("in.{}", p.name);
            s = s.replace(from, &to);
        }
    }
    s
}

pub(crate) fn emit_vertex_msl(
    name: &str,
    params: &[ShaderParam],
    return_ty: &ShaderType,
    body_source: &str,
) -> String {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let mut param_lines = Vec::new();
    let mut attr_idx = 0u32;
    let mut buf_idx = 0u32;

    for p in params {
        if p.is_uniform {
            param_lines.push(format!(
                "    constant {}&{} [[buffer({})]]",
                p.ty.msl_name(),
                if p.name.is_empty() {
                    String::new()
                } else {
                    format!(" {}", p.name)
                },
                buf_idx
            ));
            buf_idx += 1;
        } else {
            param_lines.push(format!(
                "    {} {} [[attribute({})]]",
                p.ty.msl_name(),
                p.name,
                attr_idx
            ));
            attr_idx += 1;
        }
    }

    out.push_str(&format!(
        "vertex {} {}(\n{}\n) {{\n",
        return_ty.msl_name(),
        name,
        param_lines.join(",\n")
    ));

    let body = translate_body_to_msl_shader(body_source, params);
    let body = indent_and_return_msl(&body);
    out.push_str(&body);

    out.push_str("}\n");
    out
}

pub(crate) fn emit_fragment_msl(
    name: &str,
    params: &[ShaderParam],
    return_ty: &ShaderType,
    body_source: &str,
) -> String {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let stage_in_params: Vec<&ShaderParam> = params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&ShaderParam> = params.iter().filter(|p| p.is_uniform).collect();

    if !stage_in_params.is_empty() {
        out.push_str(&format!("struct {}_Input {{\n", name));
        for (i, p) in stage_in_params.iter().enumerate() {
            out.push_str(&format!(
                "    {} {} [[user(loc{})]];\n",
                p.ty.msl_name(),
                p.name,
                i
            ));
        }
        out.push_str("};\n\n");
    }

    let mut param_lines = Vec::new();
    if !stage_in_params.is_empty() {
        param_lines.push(format!("    {}_Input in [[stage_in]]", name));
    }
    for (i, p) in uniform_params.iter().enumerate() {
        param_lines.push(format!(
            "    constant {}&{} [[buffer({})]]",
            p.ty.msl_name(),
            if p.name.is_empty() {
                String::new()
            } else {
                format!(" {}", p.name)
            },
            i
        ));
    }

    out.push_str(&format!(
        "fragment {} {}(\n{}\n) {{\n",
        return_ty.msl_name(),
        name,
        param_lines.join(",\n")
    ));

    for p in &stage_in_params {
        out.push_str(&format!(
            "    {} {} = in.{};\n",
            p.ty.msl_name(),
            p.name,
            p.name
        ));
    }

    let body = translate_body_to_msl_shader(body_source, params);
    let body = indent_and_return_msl(&body);
    out.push_str(&body);

    out.push_str("}\n");
    out
}

/// Indent body lines and convert a trailing expression into a return statement.
fn indent_and_return_msl(body: &str) -> String {
    let lines: Vec<&str> = body.trim().lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if i == lines.len() - 1 && !trimmed.is_empty() {
            if !trimmed.ends_with(';') && !trimmed.starts_with("return") {
                out.push_str(&format!("    return {};\n", trimmed));
            } else if !trimmed.starts_with("return") && !trimmed.contains("return ") {
                let without_semi = trimmed.trim_end_matches(';').trim();
                if !without_semi.contains('=')
                    && !without_semi.starts_with("auto ")
                    && !without_semi.starts_with("if ")
                    && !without_semi.starts_with("for ")
                {
                    out.push_str(&format!("    return {};\n", without_semi));
                } else {
                    out.push_str(&format!("    {}\n", trimmed));
                }
            } else {
                out.push_str(&format!("    {}\n", trimmed));
            }
        } else {
            out.push_str(&format!("    {}\n", trimmed));
        }
    }
    out
}

pub(crate) fn emit_vertex_wgsl(
    name: &str,
    params: &[ShaderParam],
    return_ty: &ShaderType,
    body_source: &str,
) -> String {
    let mut out = String::new();

    let attr_params: Vec<&ShaderParam> = params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&ShaderParam> = params.iter().filter(|p| p.is_uniform).collect();

    for (i, p) in uniform_params.iter().enumerate() {
        out.push_str(&format!(
            "@group(0) @binding({}) var<uniform> {}: {};\n",
            i,
            p.name,
            p.ty.wgsl_name()
        ));
    }
    if !uniform_params.is_empty() {
        out.push('\n');
    }

    if !attr_params.is_empty() {
        out.push_str("struct VertexInput {\n");
        for (i, p) in attr_params.iter().enumerate() {
            out.push_str(&format!(
                "    @location({}) {}: {},\n",
                i,
                p.name,
                p.ty.wgsl_name()
            ));
        }
        out.push_str("};\n\n");
    }

    out.push_str(&format!(
        "@vertex\nfn {}(in: VertexInput) -> @builtin(position) {} {{\n",
        name,
        return_ty.wgsl_name()
    ));

    let body = translate_body_to_wgsl_shader(body_source, params, true);
    let body = indent_and_return_wgsl(&body);
    out.push_str(&body);

    out.push_str("}\n");
    out
}

pub(crate) fn emit_fragment_wgsl(
    name: &str,
    params: &[ShaderParam],
    return_ty: &ShaderType,
    body_source: &str,
) -> String {
    let mut out = String::new();

    let stage_in_params: Vec<&ShaderParam> = params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&ShaderParam> = params.iter().filter(|p| p.is_uniform).collect();

    for (i, p) in uniform_params.iter().enumerate() {
        out.push_str(&format!(
            "@group(0) @binding({}) var<uniform> {}: {};\n",
            i,
            p.name,
            p.ty.wgsl_name()
        ));
    }
    if !uniform_params.is_empty() {
        out.push('\n');
    }

    if !stage_in_params.is_empty() {
        out.push_str("struct FragmentInput {\n");
        for (i, p) in stage_in_params.iter().enumerate() {
            out.push_str(&format!(
                "    @location({}) {}: {},\n",
                i,
                p.name,
                p.ty.wgsl_name()
            ));
        }
        out.push_str("};\n\n");
    }

    out.push_str(&format!(
        "@fragment\nfn {}(in: FragmentInput) -> @location(0) {} {{\n",
        name,
        return_ty.wgsl_name()
    ));

    let body = translate_body_to_wgsl_shader(body_source, params, false);
    let body = indent_and_return_wgsl(&body);
    out.push_str(&body);

    out.push_str("}\n");
    out
}

/// Indent body lines and convert trailing expression to return (WGSL).
fn indent_and_return_wgsl(body: &str) -> String {
    let lines: Vec<&str> = body.trim().lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if i == lines.len() - 1 && !trimmed.is_empty() {
            if !trimmed.ends_with(';') && !trimmed.starts_with("return") {
                out.push_str(&format!("    return {};\n", trimmed));
            } else if !trimmed.starts_with("return") && !trimmed.contains("return ") {
                let without_semi = trimmed.trim_end_matches(';').trim();
                if !without_semi.contains('=')
                    && !without_semi.starts_with("var ")
                    && !without_semi.starts_with("let ")
                    && !without_semi.starts_with("if ")
                    && !without_semi.starts_with("for ")
                {
                    out.push_str(&format!("    return {};\n", without_semi));
                } else {
                    out.push_str(&format!("    {}\n", trimmed));
                }
            } else {
                out.push_str(&format!("    {}\n", trimmed));
            }
        } else {
            out.push_str(&format!("    {}\n", trimmed));
        }
    }
    out
}
