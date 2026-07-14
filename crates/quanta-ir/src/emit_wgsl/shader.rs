//! Vertex/fragment shader WGSL emitters.

use crate::*;

/// The maximum number of combined uniform + slice storage-buffer params.
/// Texture bindings begin at 8, so at most 8 uniform/slice params fit in
/// bindings 0-7 before they collide with textures — identical to the SPIR-V
/// (`emit_spirv::MAX_SSBO_PARAMS`) and MSL (`emit_msl::shader::MAX_SSBO_PARAMS`)
/// caps, with the same error wording.
const MAX_SSBO_PARAMS: usize = 8;

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

/// The `array<ELEM>` element spelling for a `&[T]` slice param. Slice element
/// types are validated to f32/Vec2/Vec4 at DSL parse time (mirrors the MSL
/// `shader_slice_ptr_msl` float/float2/float4 set and the SPIR-V runtime-array
/// element), so anything else is treated as a `vec4<f32>` element defensively.
fn shader_slice_elem_wgsl(ty: ShaderType) -> &'static str {
    match ty {
        ShaderType::F32 => "f32",
        ShaderType::Vec2 => "vec2<f32>",
        _ => "vec4<f32>",
    }
}

/// The `@group(0) @binding(N)` index for each uniform and slice param, drawn
/// from ONE shared decl-index space (walking `params` in order, each uniform OR
/// slice consumes the next index) — identical to the SPIR-V binding, the MSL
/// `[[buffer(N)]]` index, and the runtime's `.uniform(slot, …)`. Returns the
/// binding index per param, or `None` for value attributes; also enforces the
/// combined SSBO cap with the same error the other two emitters use.
fn shared_binding_indices(shader: &ShaderDef) -> Result<Vec<Option<u32>>, String> {
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

/// Emit one read-only runtime-sized storage buffer per `&[T]` slice param, at
/// its shared decl-index binding — `@group(0) @binding(slot) var<storage, read>
/// name: array<ELEM>;`. This mirrors the MSL `const device T*` slice param and
/// the SPIR-V read-only runtime-array storage block: same slot→binding mapping,
/// same read-only semantics, same element-type set. `bindings` is the
/// shared-index table from [`shared_binding_indices`].
fn emit_slice_storage_bindings(out: &mut String, shader: &ShaderDef, bindings: &[Option<u32>]) {
    let mut any = false;
    for (p, binding) in shader.params.iter().zip(bindings.iter()) {
        if p.is_slice {
            let binding = binding.expect("a slice param always has a shared binding index");
            out.push_str(&format!(
                "@group(0) @binding({}) var<storage, read> {}: array<{}>;\n",
                binding,
                p.name,
                shader_slice_elem_wgsl(p.ty),
            ));
            any = true;
        }
    }
    if any {
        out.push('\n');
    }
}

/// Body-level source translation shared by both stages.
///
/// The body is a string-replace pass (the WGSL twin re-parses no `syn` AST):
/// constructor names are aliased to their WGSL builtins, `let mut` becomes
/// `var`, and each `&[T]` slice param's `name[index]` access is rewritten to a
/// `u32`-indexed array access `name[u32(index)]` — WGSL array indices must be
/// integral, mirroring the MSL `name[(uint)(index)]` truncation. Bounds are
/// UNCHECKED (the GPU storage-buffer contract), exactly as MSL and SPIR-V.
fn translate_shader_body_wgsl(src: &str, slice_names: &[&str]) -> String {
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
    for name in slice_names {
        s = rewrite_slice_index(&s, name);
    }
    s
}

/// Rewrite `name [ index ]` → `name[u32(index)]` for a single slice param
/// `name`, tolerating whitespace between the identifier, `[`, the index, and
/// `]`. The index expression can be any nesting-balanced sub-string (a literal,
/// an identifier, or arithmetic) up to the matching `]`; it is wrapped in
/// `u32(…)` so a computed `f32` index (`uv.x * 4.0`) truncates like the MSL
/// `(uint)(index)`. Only bracket accesses on THIS slice name are touched, so a
/// non-slice `[` elsewhere in the body is left untranslated (and would still be
/// caught downstream) — the same targeted-rewrite discipline the MSL texture
/// `sample(N` rewrite uses.
fn rewrite_slice_index(body: &str, name: &str) -> String {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < bytes.len() {
        // A slice access begins with the param name as a whole identifier: the
        // preceding byte (if any) must not be an identifier char, so `xstops`
        // never matches `stops`.
        let name_here = body[i..].starts_with(name)
            && !out.chars().next_back().is_some_and(is_ident_char)
            && body[i + name.len()..]
                .chars()
                .next()
                .is_none_or(|c| !is_ident_char(c));
        if name_here {
            // Skip whitespace after the name; the next non-space must be `[`.
            let mut j = i + name.len();
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'[' {
                // Consume the bracket body up to the matching `]`, tracking
                // nesting so an inner `[` (unlikely in a shader body, but safe)
                // does not close early.
                let mut depth = 1i32;
                let mut k = j + 1;
                while k < bytes.len() && depth > 0 {
                    match bytes[k] {
                        b'[' => depth += 1,
                        b']' => depth -= 1,
                        _ => {}
                    }
                    if depth == 0 {
                        break;
                    }
                    k += 1;
                }
                if depth == 0 {
                    let index = body[j + 1..k].trim();
                    out.push_str(name);
                    out.push_str("[u32(");
                    out.push_str(index);
                    out.push_str(")]");
                    i = k + 1;
                    continue;
                }
            }
            // Not a `name [ … ]` access — copy the name and move on.
            out.push_str(name);
            i += name.len();
            continue;
        }
        let ch = body[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The names of a shader's `&[T]` slice params, in declaration order.
fn slice_names(shader: &ShaderDef) -> Vec<&str> {
    shader
        .params
        .iter()
        .filter(|p| p.is_slice)
        .map(|p| p.name.as_str())
        .collect()
}

pub fn emit_vertex_shader(shader: &ShaderDef) -> Result<String, String> {
    let mut out = String::new();

    let bindings = shared_binding_indices(shader)?;
    // Attributes are the plain value params (neither uniform nor slice); a
    // slice binds a storage buffer, not a vertex attribute.
    let attr_params: Vec<&ShaderParam> = shader
        .params
        .iter()
        .filter(|p| !p.is_uniform && !p.is_slice)
        .collect();
    let varying_params: Vec<&ShaderParam> = attr_params.iter().skip(1).copied().collect();
    let slices = slice_names(shader);

    // Slice storage bindings precede the interface structs, at module scope.
    emit_slice_storage_bindings(&mut out, shader, &bindings);

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

    let body = translate_shader_body_wgsl(&shader.body_source, &slices);
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

    let bindings = shared_binding_indices(shader)?;
    // Interpolated inputs are the plain value params (neither uniform nor
    // slice); a slice binds a storage buffer, not a stage input.
    let stage_in_params: Vec<&ShaderParam> = shader
        .params
        .iter()
        .filter(|p| !p.is_uniform && !p.is_slice)
        .collect();
    let slices = slice_names(shader);

    // Slice storage bindings precede the interface struct, at module scope.
    emit_slice_storage_bindings(&mut out, shader, &bindings);

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

    let body = translate_shader_body_wgsl(&shader.body_source, &slices);
    let trimmed = body.trim();
    out.push_str(&format!("    return {};\n", trimmed));
    out.push_str("}\n");

    Ok(out)
}
