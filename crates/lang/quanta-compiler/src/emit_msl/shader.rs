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
use super::shader_ast::{BodyTail, MslType};

/// The maximum number of combined uniform + slice storage-buffer params. Metal
/// textures/samplers occupy their own index space, but the shared decl-index
/// mirrors the Vulkan binding contract, where textures start at binding 8.
const MAX_SSBO_PARAMS: usize = 8;

/// Seed the body emitter's type environment with each param's MSL type, so a
/// `let x = if ...` whose value flows from a param can name its declared type,
/// and so a `name[index]` on a slice param can be validated.
fn shader_param_types(shader: &quanta_ir::ShaderDef) -> Vec<(String, MslType)> {
    shader
        .params
        .iter()
        .map(|p| {
            let ty = if p.is_slice {
                MslType::slice_of(p.ty)
            } else {
                MslType::from_shader_type(p.ty)
            };
            (p.name.clone(), ty)
        })
        .collect()
}

/// The `[[buffer(N)]]` index for each uniform and slice param, drawn from ONE
/// shared decl-index space (walking `params` in order, each uniform OR slice
/// consumes the next index) — identical to the SPIR-V binding and the runtime's
/// `.uniform(slot, …)`. Returns the buffer index per param, or `None` for value
/// attributes; also enforces the combined SSBO cap.
fn shared_buffer_indices(shader: &quanta_ir::ShaderDef) -> Result<Vec<Option<u32>>, String> {
    let combined = shader
        .params
        .iter()
        .filter(|p| p.is_uniform || p.is_slice)
        .count();
    if combined > MAX_SSBO_PARAMS {
        return Err(format!(
            "shader `{}` declares {combined} combined uniform+slice params, over the \
             cap of {MAX_SSBO_PARAMS} (texture bindings start at 8)",
            shader.name
        ));
    }
    let mut out = Vec::with_capacity(shader.params.len());
    let mut next = 0u32;
    for p in &shader.params {
        if p.is_uniform || p.is_slice {
            out.push(Some(next));
            next += 1;
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

/// Whether `body_source` samples texture slot `slot`, tolerating whitespace
/// between `sample`, `(`, and the slot digit (`sample ( 0`, `sample( 0`, …).
/// Any non-macro `ShaderDef` producer — or a future printer change — could
/// space these apart, so the scan must not depend on a contiguous `sample(N`.
fn body_samples_slot(body: &str, slot: u32) -> bool {
    let digit = char::from_digit(slot, 10).unwrap();
    let bytes = body.as_bytes();
    let mut i = 0;
    while let Some(rel) = body[i..].find("sample") {
        let mut j = i + rel + "sample".len();
        // optional whitespace, then '('
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j < bytes.len() && bytes[j] == b'(' {
            j += 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == digit as u8 {
                return true;
            }
        }
        i += rel + "sample".len();
    }
    false
}

/// Whether `body` calls the argument-free builtin `name` (`frag_coord`,
/// `vertex_id`, `instance_id`), tolerating whitespace between the name and
/// `(` (the same scan contract as `body_samples_slot`). Only the call form
/// counts: the DSL has no user-defined functions, so an identifier followed
/// by `(` can only be a builtin call, and a param whose NAME contains the
/// substring is never followed by `(`. The SPIR-V tree carries its own copy
/// (`emit_spirv::body_calls`), like `body_samples_slot`.
fn body_calls(body: &str, name: &str) -> bool {
    let bytes = body.as_bytes();
    let mut i = 0;
    while let Some(rel) = body[i..].find(name) {
        let mut j = i + rel + name.len();
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j < bytes.len() && bytes[j] == b'(' {
            return true;
        }
        i += rel + name.len();
    }
    false
}

/// The `const device T*` pointer spelling for a `&[T]` slice param.
fn shader_slice_ptr_msl(ty: quanta_ir::ShaderType) -> &'static str {
    match ty {
        quanta_ir::ShaderType::F32 => "float",
        quanta_ir::ShaderType::Vec2 => "float2",
        // Slice element types are validated to f32/Vec2/Vec4 at parse time.
        _ => "float4",
    }
}

fn shader_type_msl(ty: quanta_ir::ShaderType) -> &'static str {
    match ty {
        quanta_ir::ShaderType::F32 => "float",
        quanta_ir::ShaderType::Vec2 => "float2",
        quanta_ir::ShaderType::Vec3 => "float3",
        quanta_ir::ShaderType::Vec4 => "float4",
        quanta_ir::ShaderType::Mat4 => "float4x4",
        quanta_ir::ShaderType::Mat3 => "float3x3",
        quanta_ir::ShaderType::U32 => "uint",
    }
}

/// The interpolation qualifier a varying member of type `ty` carries in the
/// vertex-out / fragment-in structs. Integer varyings MUST be `[[flat]]` —
/// Metal cannot interpolate integers and rejects the pipeline otherwise
/// (the MSL twin of the SPIR-V `Flat` decoration on both interface ends).
/// Float varyings keep default (perspective-correct) interpolation. Vertex
/// ATTRIBUTES take no qualifier — they are fetched, not interpolated.
fn varying_qualifier_msl(ty: quanta_ir::ShaderType) -> &'static str {
    match ty {
        quanta_ir::ShaderType::U32 => " [[flat]]",
        _ => "",
    }
}

/// Emit the shared Varyings interface struct — used as the vertex stage-out
/// AND the fragment stage-in, under the user's own struct name. The
/// `#[position]` field is the `[[position]]` member (always first); each
/// varying takes `[[user(locN)]]` at its field-declaration-order Location
/// (`[[flat]]` for integers), matching the SPIR-V Location/Flat decorations
/// byte for byte.
fn emit_varyings_struct_msl(out: &mut String, v: &quanta_ir::ShaderVaryings) {
    out.push_str(&format!("struct {} {{\n", v.struct_name));
    out.push_str(&format!("    float4 {} [[position]];\n", v.position));
    for (i, f) in v.fields.iter().enumerate() {
        out.push_str(&format!(
            "    {} {} [[user(loc{})]]{};\n",
            shader_type_msl(f.ty),
            f.name,
            i,
            varying_qualifier_msl(f.ty),
        ));
    }
    out.push_str("};\n\n");
}

/// A fragment `ShaderDef` may not declare plain value params: fragment stage
/// inputs come from the shared `#[derive(Varyings)]` struct (read as
/// `<receiver>.<field>` in the body). Structural rejection with the same
/// wording as the SPIR-V and WGSL emitters.
fn reject_fragment_value_params(shader: &quanta_ir::ShaderDef) -> Result<(), String> {
    match shader.params.iter().find(|p| !p.is_uniform && !p.is_slice) {
        Some(p) => Err(format!(
            "fragment shader `{}` declares value param `{}`: fragment stage inputs \
             come from the #[derive(Varyings)] struct",
            shader.name, p.name
        )),
        None => Ok(()),
    }
}

/// Emit MSL for a vertex shader.
///
/// Metal requires vertex attributes to be passed via a struct with `[[stage_in]]`.
/// Uniform parameters use `[[buffer(N)]]` bindings.
pub fn emit_vertex_shader(shader: &quanta_ir::ShaderDef) -> Result<String, String> {
    // `frag_coord()` is fragment-only: without this guard the AST walker
    // would lower the call to the `_frag_coord` identifier, which no vertex
    // signature declares — invalid MSL that only fails at metallib compile.
    // Reject structurally instead (the SPIR-V vertex path errors in its body
    // parser and falls to a passthrough — the same not-accepted verdict).
    if body_calls(&shader.body_source, "frag_coord") {
        return Err(format!(
            "vertex shader `{}` calls frag_coord(), which is only available in fragment shaders",
            shader.name
        ));
    }

    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let buffer_indices = shared_buffer_indices(shader)?;
    // Attributes are the plain value params (neither uniform nor slice).
    let attr_params: Vec<&quanta_ir::ShaderParam> = shader
        .params
        .iter()
        .filter(|p| !p.is_uniform && !p.is_slice)
        .collect();
    // Uniform + slice params, each paired with its shared buffer index.
    let ssbo_params: Vec<(&quanta_ir::ShaderParam, u32)> = shader
        .params
        .iter()
        .zip(buffer_indices.iter())
        .filter_map(|(p, b)| b.map(|b| (p, b)))
        .collect();

    // Emit vertex input struct with [[attribute(N)]] decorations. Attributes
    // are PURE inputs — nothing is auto-forwarded; every varying is written
    // explicitly through the Varyings struct literal in the body tail.
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

    // The stage-out struct: the shared Varyings struct under the user's own
    // name, or a position-only `{name}_VertexOut` when the shader has no
    // varying interface (`-> Vec4`).
    let out_struct = match &shader.varyings {
        Some(v) => {
            emit_varyings_struct_msl(&mut out, v);
            v.struct_name.clone()
        }
        None => {
            out.push_str(&format!("struct {}_VertexOut {{\n", shader.name));
            out.push_str(&format!(
                "    {} position [[position]];\n",
                shader_type_msl(shader.return_type),
            ));
            out.push_str("};\n\n");
            format!("{}_VertexOut", shader.name)
        }
    };

    // Build parameter list
    let mut param_lines = Vec::new();
    if !attr_params.is_empty() {
        param_lines.push(format!("    {}_VertexIn in [[stage_in]]", shader.name));
    }
    // Vertex-index builtins: each declared only when the body calls it
    // (whitespace-tolerant scan, like the texture slots). The AST walker
    // lowers `vertex_id()` / `instance_id()` to these exact identifiers —
    // see `emit_call` in `shader_ast.rs`.
    if body_calls(&shader.body_source, "vertex_id") {
        param_lines.push("    uint _vertex_id [[vertex_id]]".to_string());
    }
    if body_calls(&shader.body_source, "instance_id") {
        param_lines.push("    uint _instance_id [[instance_id]]".to_string());
    }
    for (p, buffer) in &ssbo_params {
        if p.is_slice {
            param_lines.push(format!(
                "    const device {}* {} [[buffer({})]]",
                shader_slice_ptr_msl(p.ty),
                p.name,
                buffer,
            ));
        } else {
            param_lines.push(format!(
                "    constant {}& {} [[buffer({})]]",
                shader_type_msl(p.ty),
                p.name,
                buffer,
            ));
        }
    }

    out.push_str(&format!(
        "vertex {} {}(\n{}\n) {{\n",
        out_struct,
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

    let param_types = shader_param_types(shader);
    if let Some(v) = &shader.varyings {
        // Shared-struct model: declare the stage-out struct up front; the
        // body's tail struct literal lowers to one member assignment per
        // field (position included) — nothing is forwarded implicitly.
        out.push_str(&format!("    {} out;\n", v.struct_name));
        let body = shader_ast::emit_body(
            &shader.body_source,
            BodyTail::StructOut("out"),
            &param_types,
            Some(v),
        )?;
        out.push_str(&body);
        out.push_str("    return out;\n");
        out.push_str("}\n");
        return Ok(out);
    }

    // Position-only vertex: declare the position result, then lower the body
    // to assign it — the vertex tail IS the clip-space position.
    out.push_str(&format!(
        "    {} pos_result;\n",
        shader_type_msl(shader.return_type),
    ));
    let body = shader_ast::emit_body(
        &shader.body_source,
        BodyTail::Assign("pos_result"),
        &param_types,
        None,
    )?;
    out.push_str(&body);

    out.push_str(&format!("    {out_struct} out;\n"));
    out.push_str("    out.position = pos_result;\n");
    out.push_str("    return out;\n");
    out.push_str("}\n");
    Ok(out)
}

/// Emit MSL for a fragment shader.
pub fn emit_fragment_shader(shader: &quanta_ir::ShaderDef) -> Result<String, String> {
    // `vertex_id()` / `instance_id()` are vertex-only: without this guard
    // the AST walker would lower the call to the `_vertex_id` /
    // `_instance_id` identifier, which no fragment signature declares —
    // invalid MSL that only fails at metallib compile. Reject structurally
    // instead (the SPIR-V fragment path errors in its body parser and falls
    // to a passthrough — the same not-accepted verdict), mirroring the
    // vertex emitter's `frag_coord()` rejection with the polarity flipped.
    for builtin in ["vertex_id", "instance_id"] {
        if body_calls(&shader.body_source, builtin) {
            return Err(format!(
                "fragment shader `{}` calls {builtin}(), which is only available in vertex shaders",
                shader.name
            ));
        }
    }

    // A fragment's stage inputs come from the Varyings struct — a plain
    // value param has no meaning here anymore (same structural rejection as
    // the SPIR-V and WGSL emitters).
    reject_fragment_value_params(shader)?;

    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let buffer_indices = shared_buffer_indices(shader)?;
    let ssbo_params: Vec<(&quanta_ir::ShaderParam, u32)> = shader
        .params
        .iter()
        .zip(buffer_indices.iter())
        .filter_map(|(p, b)| b.map(|b| (p, b)))
        .collect();

    // Stage-in struct: the shared Varyings struct under the user's own name —
    // the same member list the vertex stage-out declares ([[user(locN)]] /
    // [[flat]] matching), with the `[[position]]` member carrying the
    // interpolated window position for `<receiver>.<position>` reads.
    if let Some(v) = &shader.varyings {
        emit_varyings_struct_msl(&mut out, v);
    }

    // Detect texture slots used in body (whitespace-tolerant `sample(N` scan).
    let max_tex_slot = (0..8u32)
        .filter(|slot| body_samples_slot(&shader.body_source, *slot))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut param_lines = Vec::new();
    if let Some(v) = &shader.varyings {
        let recv = v.binding.as_deref().ok_or_else(|| {
            format!(
                "fragment shader `{}`: the varyings interface names no receiver param",
                shader.name
            )
        })?;
        param_lines.push(format!("    {} {recv} [[stage_in]]", v.struct_name));
    }
    // Window-space position builtin: declared only when the body calls
    // `frag_coord()` (whitespace-tolerant scan, like the texture slots). The
    // AST walker lowers the call to this exact identifier — see `emit_call`
    // in `shader_ast.rs`.
    if body_calls(&shader.body_source, "frag_coord") {
        param_lines.push("    float4 _frag_coord [[position]]".to_string());
    }
    for (p, buffer) in &ssbo_params {
        if p.is_slice {
            param_lines.push(format!(
                "    const device {}* {} [[buffer({})]]",
                shader_slice_ptr_msl(p.ty),
                p.name,
                buffer,
            ));
        } else {
            param_lines.push(format!(
                "    constant {}& {} [[buffer({})]]",
                shader_type_msl(p.ty),
                p.name,
                buffer,
            ));
        }
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

    // Lower the body: the fragment tail is the output color, so route it
    // through `return`. Varyings are read directly as `<receiver>.<field>` —
    // no unpacking. `sample(slot, uv)` is emitted verbatim by the AST
    // walker, then rewritten to `tex_N.sample(smp_N, uv)` here — a targeted
    // rewrite on the already-structured MSL. The AST walker normalizes the slot
    // to a float literal (`sample(0.0, uv)`), but the rewrite tolerates
    // whitespace between `sample`, `(`, and the digit so it never depends on a
    // contiguous form.
    let param_types = shader_param_types(shader);
    let mut body = shader_ast::emit_body(
        &shader.body_source,
        BodyTail::Return,
        &param_types,
        shader.varyings.as_ref(),
    )?;
    for slot in 0..max_tex_slot {
        body = rewrite_sample_slot(&body, slot);
    }
    out.push_str(&body);
    out.push_str("}\n");
    Ok(out)
}

/// Rewrite `sample ( N[.0] ,` → `tex_N.sample(smp_N,` in an emitted MSL body,
/// tolerating whitespace between `sample`, `(`, and the slot digit. The AST
/// walker emits the contiguous `sample(N.0,` form today, but any other producer
/// (or a printer change) must not break the texture binding silently.
fn rewrite_sample_slot(body: &str, slot: u32) -> String {
    let digit = char::from_digit(slot, 10).unwrap();
    let replacement = format!("tex_{slot}.sample(smp_{slot},");
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < bytes.len() {
        if body[i..].starts_with("sample") {
            let skip_ws = |mut k: usize| {
                while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                    k += 1;
                }
                k
            };
            let after_sample = skip_ws(i + "sample".len());
            if after_sample < bytes.len() && bytes[after_sample] == b'(' {
                let after_paren = skip_ws(after_sample + 1);
                if after_paren < bytes.len() && bytes[after_paren] == digit as u8 {
                    // Consume the digit and an optional `.0` float suffix.
                    let mut k = after_paren + 1;
                    if body[k..].starts_with(".0") {
                        k += 2;
                    }
                    let after_num = skip_ws(k);
                    if after_num < bytes.len() && bytes[after_num] == b',' {
                        out.push_str(&replacement);
                        i = after_num + 1;
                        continue;
                    }
                }
            }
            // Not a `sample(N,` at this position — copy `sample` and advance.
            out.push_str("sample");
            i += "sample".len();
            continue;
        }
        // Copy one UTF-8 char.
        let ch = body[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

#[cfg(test)]
#[path = "shader_tests.rs"]
mod tests;
