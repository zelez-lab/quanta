//! Vertex/fragment shader MSL emitters.

fn shader_type_msl(ty: crate::ShaderType) -> &'static str {
    match ty {
        crate::ShaderType::F32 => "float",
        crate::ShaderType::Vec2 => "float2",
        crate::ShaderType::Vec3 => "float3",
        crate::ShaderType::Vec4 => "float4",
        crate::ShaderType::Mat4 => "float4x4",
        crate::ShaderType::Mat3 => "float3x3",
        crate::ShaderType::U32 => "uint",
    }
}

/// Translate a Rust-like shader body to MSL (basic string substitutions).
///
/// Handles both hand-written source and tokenized source (proc_macro2
/// tokenizes `Vec4::new` as `Vec4 :: new` with spaces around `::`) .
fn translate_shader_body(src: &str) -> String {
    let mut s = src.to_string();
    // Strip outer braces from block expression
    let trimmed = s.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        s = trimmed[1..trimmed.len() - 1].to_string();
    }
    // Handle tokenized form (spaces around ::)
    s = s.replace("Vec4 :: new(", "float4(");
    s = s.replace("Vec3 :: new(", "float3(");
    s = s.replace("Vec2 :: new(", "float2(");
    // Handle direct source form
    s = s.replace("Vec4::new(", "float4(");
    s = s.replace("Vec3::new(", "float3(");
    s = s.replace("Vec2::new(", "float2(");
    s = s.replace("let mut ", "auto ");
    s = s.replace("let ", "auto ");
    s
}

/// Wrap the last expression of a body as a return statement.
fn indent_and_return(body: &str) -> String {
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
            } else if !trimmed.contains("return") {
                let without_semi = trimmed.trim_end_matches(';').trim();
                if !without_semi.contains('=') {
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

/// Emit MSL for a vertex shader.
///
/// Metal requires vertex attributes to be passed via a struct with `[[stage_in]]`.
/// Uniform parameters use `[[buffer(N)]]` bindings.
pub fn emit_vertex_shader(shader: &crate::ShaderDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let attr_params: Vec<&crate::ShaderParam> =
        shader.params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&crate::ShaderParam> =
        shader.params.iter().filter(|p| p.is_uniform).collect();

    // Emit vertex input struct with [[attribute(N)]] decorations
    if !attr_params.is_empty() {
        out.push_str(&format!("struct {}_VertexIn {{\n", shader.name));
        for (i, p) in attr_params.iter().enumerate() {
            out.push_str(&format!(
                "    {} {} [[attribute({})]];\n",
                shader_type_msl(p.ty),
                p.name,
                i,
            ));
        }
        out.push_str("};\n\n");
    }

    // Build parameter list
    let mut param_lines = Vec::new();
    if !attr_params.is_empty() {
        param_lines.push(format!("    {}_VertexIn in [[stage_in]]", shader.name));
    }
    for (i, p) in uniform_params.iter().enumerate() {
        param_lines.push(format!(
            "    constant {}& {} [[buffer({})]]",
            shader_type_msl(p.ty),
            p.name,
            i,
        ));
    }

    out.push_str(&format!(
        "vertex {} {}(\n{}\n) {{\n",
        shader_type_msl(shader.return_type),
        shader.name,
        param_lines.join(",\n"),
    ));

    // Unpack struct members into local variables
    for p in &attr_params {
        out.push_str(&format!(
            "    {} {} = in.{};\n",
            shader_type_msl(p.ty),
            p.name,
            p.name,
        ));
    }

    let body = translate_shader_body(&shader.body_source);
    out.push_str(&indent_and_return(&body));
    out.push_str("}\n");
    Ok(out)
}

/// Emit MSL for a fragment shader.
pub fn emit_fragment_shader(shader: &crate::ShaderDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let stage_in_params: Vec<&crate::ShaderParam> =
        shader.params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&crate::ShaderParam> =
        shader.params.iter().filter(|p| p.is_uniform).collect();

    // Stage-in struct for interpolated inputs. Integer members must be
    // `[[flat]]` — Metal cannot interpolate integers.
    if !stage_in_params.is_empty() {
        out.push_str(&format!("struct {}_Input {{\n", shader.name));
        for (i, p) in stage_in_params.iter().enumerate() {
            let flat = if p.ty == crate::ShaderType::U32 {
                " [[flat]]"
            } else {
                ""
            };
            out.push_str(&format!(
                "    {} {} [[user(loc{})]]{};\n",
                shader_type_msl(p.ty),
                p.name,
                i,
                flat,
            ));
        }
        out.push_str("};\n\n");
    }

    let mut param_lines = Vec::new();
    if !stage_in_params.is_empty() {
        param_lines.push(format!("    {}_Input in [[stage_in]]", shader.name));
    }
    for (i, p) in uniform_params.iter().enumerate() {
        param_lines.push(format!(
            "    constant {}& {} [[buffer({})]]",
            shader_type_msl(p.ty),
            p.name,
            i,
        ));
    }

    out.push_str(&format!(
        "fragment {} {}(\n{}\n) {{\n",
        shader_type_msl(shader.return_type),
        shader.name,
        param_lines.join(",\n"),
    ));

    // Unpack stage_in members
    for p in &stage_in_params {
        out.push_str(&format!(
            "    {} {} = in.{};\n",
            shader_type_msl(p.ty),
            p.name,
            p.name,
        ));
    }

    let body = translate_shader_body(&shader.body_source);
    out.push_str(&indent_and_return(&body));
    out.push_str("}\n");
    Ok(out)
}
