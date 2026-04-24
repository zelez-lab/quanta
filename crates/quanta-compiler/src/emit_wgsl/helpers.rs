//! Helper functions for WGSL emission.

use quanta_ir::*;

pub(crate) fn const_wgsl(v: &ConstValue) -> String {
    match v {
        ConstValue::F32(x) => format!("{}f", x),
        ConstValue::U32(x) => format!("{}u", x),
        ConstValue::I32(x) => format!("{}i", x),
        ConstValue::Bool(x) => format!("{}", x),
        _ => "/* unsupported const */".to_string(),
    }
}

/// Translate a Rust device function source to WGSL.
/// WGSL uses `fn name(...) -> type` — same syntax as Rust for function
/// signatures, so only body-level translations are needed.
pub(crate) fn translate_device_fn_to_wgsl(rust_source: &str) -> String {
    let mut s = rust_source.to_string();
    s = s.replace("let mut ", "var ");
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s
}

// ── Vertex/Fragment WGSL emitters ──────────────────────────────────────────

fn shader_type_wgsl(ty: ShaderType) -> &'static str {
    match ty {
        ShaderType::F32 => "f32",
        ShaderType::Vec2 => "vec2<f32>",
        ShaderType::Vec3 => "vec3<f32>",
        ShaderType::Vec4 => "vec4<f32>",
        ShaderType::Mat4 => "mat4x4<f32>",
        ShaderType::Mat3 => "mat3x3<f32>",
    }
}

fn translate_shader_body_wgsl(src: &str) -> String {
    let mut s = src.trim().to_string();
    if s.starts_with('{') && s.ends_with('}') {
        s = s[1..s.len() - 1].to_string();
    }
    s = s.replace("Vec4 :: new(", "vec4<f32>(");
    s = s.replace("Vec4 :: new (", "vec4<f32>(");
    s = s.replace("Vec4::new(", "vec4<f32>(");
    s = s.replace("Vec3 :: new(", "vec3<f32>(");
    s = s.replace("Vec3::new(", "vec3<f32>(");
    s = s.replace("Vec2 :: new(", "vec2<f32>(");
    s = s.replace("Vec2::new(", "vec2<f32>(");
    s = s.replace("let mut ", "var ");
    s
}

pub fn emit_vertex_shader(shader: &ShaderDef) -> Result<String, String> {
    let mut out = String::new();

    let attr_params: Vec<&ShaderParam> = shader.params.iter().filter(|p| !p.is_uniform).collect();
    let varying_params: Vec<&ShaderParam> = attr_params.iter().skip(1).copied().collect();

    // Input struct
    out.push_str("struct VertexInput {\n");
    for (i, p) in attr_params.iter().enumerate() {
        out.push_str(&format!(
            "    @location({}) {}: {},\n",
            i,
            p.name,
            shader_type_wgsl(p.ty)
        ));
    }
    out.push_str("};\n\n");

    // Output struct
    out.push_str("struct VertexOutput {\n");
    out.push_str("    @builtin(position) position: vec4<f32>,\n");
    for (i, p) in varying_params.iter().enumerate() {
        out.push_str(&format!(
            "    @location({}) {}: {},\n",
            i,
            p.name,
            shader_type_wgsl(p.ty)
        ));
    }
    out.push_str("};\n\n");

    out.push_str("@vertex\nfn main(in: VertexInput) -> VertexOutput {\n");
    for p in &attr_params {
        out.push_str(&format!("    let {} = in.{};\n", p.name, p.name));
    }

    let body = translate_shader_body_wgsl(&shader.body_source);
    let trimmed = body.trim();
    out.push_str(&format!("    let pos_result = {};\n", trimmed));
    out.push_str("    var output: VertexOutput;\n");
    out.push_str("    output.position = pos_result;\n");
    for p in &varying_params {
        out.push_str(&format!("    output.{} = {};\n", p.name, p.name));
    }
    out.push_str("    return output;\n");
    out.push_str("}\n");

    Ok(out)
}

pub fn emit_fragment_shader(shader: &ShaderDef) -> Result<String, String> {
    let mut out = String::new();

    let stage_in_params: Vec<&ShaderParam> =
        shader.params.iter().filter(|p| !p.is_uniform).collect();

    if !stage_in_params.is_empty() {
        out.push_str("struct FragmentInput {\n");
        for (i, p) in stage_in_params.iter().enumerate() {
            out.push_str(&format!(
                "    @location({}) {}: {},\n",
                i,
                p.name,
                shader_type_wgsl(p.ty)
            ));
        }
        out.push_str("};\n\n");
    }

    out.push_str("@fragment\nfn main(");
    if !stage_in_params.is_empty() {
        out.push_str("in: FragmentInput");
    }
    out.push_str(") -> @location(0) vec4<f32> {\n");

    for p in &stage_in_params {
        out.push_str(&format!("    let {} = in.{};\n", p.name, p.name));
    }

    let body = translate_shader_body_wgsl(&shader.body_source);
    let trimmed = body.trim();
    out.push_str(&format!("    return {};\n", trimmed));
    out.push_str("}\n");

    Ok(out)
}
