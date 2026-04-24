//! Vertex and fragment shader MSL emission.
//!
//! These operate on `ShaderDef` (not `KernelDef`) and produce complete
//! MSL source files for the Metal render pipeline.

fn shader_type_msl(ty: quanta_ir::ShaderType) -> &'static str {
    match ty {
        quanta_ir::ShaderType::F32 => "float",
        quanta_ir::ShaderType::Vec2 => "float2",
        quanta_ir::ShaderType::Vec3 => "float3",
        quanta_ir::ShaderType::Vec4 => "float4",
        quanta_ir::ShaderType::Mat4 => "float4x4",
        quanta_ir::ShaderType::Mat3 => "float3x3",
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

/// Wrap the last expression as an assignment to a variable (for vertex output struct).
fn indent_and_return_to_var(body: &str, var_name: &str) -> String {
    let lines: Vec<&str> = body.trim().lines().collect();
    if lines.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if i == lines.len() - 1 && !trimmed.is_empty() {
            let expr = trimmed
                .trim_end_matches(';')
                .trim()
                .trim_start_matches("return ")
                .trim();
            out.push_str(&format!("    auto {} = {};\n", var_name, expr));
        } else {
            out.push_str(&format!("    {}\n", trimmed));
        }
    }
    out
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
pub fn emit_vertex_shader(shader: &quanta_ir::ShaderDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let attr_params: Vec<&quanta_ir::ShaderParam> =
        shader.params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&quanta_ir::ShaderParam> =
        shader.params.iter().filter(|p| p.is_uniform).collect();

    // Varying params = all attr params except the first (position)
    let varying_params: Vec<&quanta_ir::ShaderParam> =
        attr_params.iter().skip(1).copied().collect();

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

    // Emit vertex output struct: [[position]] + varying members with [[user(locN)]]
    out.push_str(&format!("struct {}_VertexOut {{\n", shader.name));
    out.push_str(&format!(
        "    {} position [[position]];\n",
        shader_type_msl(shader.return_type),
    ));
    for (i, p) in varying_params.iter().enumerate() {
        out.push_str(&format!(
            "    {} {} [[user(loc{})]];\n",
            shader_type_msl(p.ty),
            p.name,
            i,
        ));
    }
    out.push_str("};\n\n");

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
        "vertex {}_VertexOut {}(\n{}\n) {{\n",
        shader.name,
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

    // Evaluate body for position
    let body = translate_shader_body(&shader.body_source);
    let body = indent_and_return_to_var(&body, "pos_result");
    out.push_str(&body);

    // Build output struct
    out.push_str(&format!("    {}_VertexOut out;\n", shader.name));
    out.push_str("    out.position = pos_result;\n");
    for p in &varying_params {
        out.push_str(&format!("    out.{} = {};\n", p.name, p.name));
    }
    out.push_str("    return out;\n");
    out.push_str("}\n");
    Ok(out)
}

/// Emit MSL for a fragment shader.
pub fn emit_fragment_shader(shader: &quanta_ir::ShaderDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let stage_in_params: Vec<&quanta_ir::ShaderParam> =
        shader.params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&quanta_ir::ShaderParam> =
        shader.params.iter().filter(|p| p.is_uniform).collect();

    // Stage-in struct for interpolated inputs
    if !stage_in_params.is_empty() {
        out.push_str(&format!("struct {}_Input {{\n", shader.name));
        for (i, p) in stage_in_params.iter().enumerate() {
            out.push_str(&format!(
                "    {} {} [[user(loc{})]];\n",
                shader_type_msl(p.ty),
                p.name,
                i,
            ));
        }
        out.push_str("};\n\n");
    }

    // Detect texture slots used in body
    let max_tex_slot = (0..8u32)
        .filter(|slot| shader.body_source.contains(&format!("sample({}", slot)))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

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
    // Add texture + sampler params for each used slot
    for slot in 0..max_tex_slot {
        param_lines.push(format!(
            "    texture2d<float> tex_{} [[texture({})]]",
            slot, slot,
        ));
        param_lines.push(format!("    sampler smp_{} [[sampler({})]]", slot, slot,));
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

    // Translate body, replacing sample(N, uv) with tex_N.sample(smp_N, uv)
    let mut body = translate_shader_body(&shader.body_source);
    for slot in 0..max_tex_slot {
        // Handle both spaced and compact forms
        let patterns = [
            format!("sample({} ,", slot),
            format!("sample({},", slot),
            format!("sample ({} ,", slot),
            format!("sample ({},", slot),
        ];
        for pat in &patterns {
            body = body.replace(pat, &format!("tex_{}.sample(smp_{},", slot, slot));
        }
    }
    out.push_str(&indent_and_return(&body));
    out.push_str("}\n");
    Ok(out)
}
