//! Vertex and fragment shader MSL emission.
//!
//! These operate on `ShaderDef` (not `KernelDef`) and produce complete
//! MSL source files for the Metal render pipeline.
//!
//! The signature/struct/binding shell (vertex-in / vertex-out structs,
//! `[[stage_in]]`, `constant T& [[buffer(n)]]` uniforms, `[[texture(n)]]` /
//! `[[sampler(n)]]` for sampled slots) is emitted here; the function BODY is
//! lowered by the AST walker in [`super::shader_ast`], which re-parses the
//! token-stringified Rust body and walks the real `syn` AST — so path
//! line-wraps (`Vec4 :: new`), statement-position `if`, and `&T` uniform
//! derefs all translate correctly. The old string-replace path miscompiled
//! all three.

use super::shader_ast;
use super::shader_ast::MslType;

/// Seed the body emitter's type environment with each param's MSL type, so a
/// `let x = if ...` whose value flows from a param can name its declared type.
fn shader_param_types(shader: &quanta_ir::ShaderDef) -> Vec<(String, MslType)> {
    shader
        .params
        .iter()
        .map(|p| (p.name.clone(), MslType::from_shader_type(p.ty)))
        .collect()
}

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

    // Declare the position result, then lower the body to assign it. The
    // vertex tail is the clip-space position; varyings are forwarded raw from
    // the inputs (below), matching the SPIR-V vertex path's varying model.
    out.push_str(&format!(
        "    {} pos_result;\n",
        shader_type_msl(shader.return_type),
    ));
    let param_types = shader_param_types(shader);
    let body = shader_ast::emit_body(&shader.body_source, Some("pos_result"), &param_types)?;
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

    // Lower the body: the fragment tail is the output color, so route it
    // through `return`. `sample(slot, uv)` is emitted verbatim by the AST
    // walker, then rewritten to `tex_N.sample(smp_N, uv)` here — a targeted
    // rewrite on the already-structured MSL (the slot is a literal, the
    // spacing is fixed), not a translation of the raw Rust source.
    let param_types = shader_param_types(shader);
    let mut body = shader_ast::emit_body(&shader.body_source, None, &param_types)?;
    for slot in 0..max_tex_slot {
        body = body.replace(
            &format!("sample({}.0,", slot),
            &format!("tex_{}.sample(smp_{},", slot, slot),
        );
    }
    out.push_str(&body);
    out.push_str("}\n");
    Ok(out)
}

#[cfg(test)]
#[path = "shader_tests.rs"]
mod tests;
