//! Vertex/fragment shader WGSL emitters.
//!
//! The interface shell (input/output structs, uniform + slice + texture
//! bindings, the stage-builtin entry-point params) is emitted here; the
//! function BODY is lowered by the hand-rolled recursive-descent walker in
//! [`super::shader_walker`], which re-tokenizes the token-stringified Rust
//! body and emits real WGSL statements — so `let`/`let mut`,
//! statement-`if`/`else`, value-`if`, bounded `for` loops, u32
//! params/varyings/literals, the stage builtins (`vertex_id()` /
//! `instance_id()` / `frag_coord()`), `&T` uniform derefs, `&[T]` slice
//! indexing, swizzles, intrinsics, and `sample(N, uv)` all translate. The
//! construct surface mirrors the SPIR-V shader walker: what SPIR-V accepts,
//! this accepts; what SPIR-V rejects, this rejects with a clear error.

use super::shader_walker::{ParamInfo, param_infos, walk_body, walk_body_varyings};
use crate::*;

/// The maximum number of combined uniform + slice storage-buffer params.
/// Texture bindings begin at 8, so at most 8 uniform/slice params fit in
/// bindings 0-7 before they collide with textures — identical to the SPIR-V
/// (`emit_spirv::MAX_SSBO_PARAMS`) and MSL (`emit_msl::shader::MAX_SSBO_PARAMS`)
/// caps, with the same error wording.
const MAX_SSBO_PARAMS: usize = 8;

/// Texture bindings begin here — past the eight uniform/slice binding slots.
const TEXTURE_BINDING_BASE: u32 = 8;

fn shader_type_wgsl(ty: ShaderType) -> &'static str {
    match ty {
        ShaderType::F32 => "f32",
        ShaderType::Vec2 => "vec2<f32>",
        ShaderType::Vec3 => "vec3<f32>",
        ShaderType::Vec4 => "vec4<f32>",
        ShaderType::Mat4 => "mat4x4<f32>",
        ShaderType::Mat3 => "mat3x3<f32>",
        ShaderType::U32 => "u32",
    }
}

/// Whether `body` calls the argument-free builtin `name` (`frag_coord`,
/// `vertex_id`, `instance_id`), tolerating whitespace between the name and
/// `(` (the same scan contract as [`body_samples_slot`]). Only the call form
/// counts: the DSL has no user-defined functions, so an identifier followed
/// by `(` can only be a builtin call, and a param whose NAME contains the
/// substring is never followed by `(`. The MSL and SPIR-V emitters carry the
/// same scan (`emit_msl::shader::body_calls` / `emit_spirv::body_calls`).
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

/// A fragment `ShaderDef` may not declare plain value params: fragment stage
/// inputs come from the shared `#[derive(Varyings)]` struct (read as
/// `<receiver>.<field>` in the body). Structural rejection with the same
/// wording as the SPIR-V and MSL emitters.
fn reject_fragment_value_params(shader: &ShaderDef) -> Result<(), String> {
    match shader.params.iter().find(|p| !p.is_uniform && !p.is_slice) {
        Some(p) => Err(format!(
            "fragment shader `{}` declares value param `{}`: fragment stage inputs \
             come from the #[derive(Varyings)] struct",
            shader.name, p.name
        )),
        None => Ok(()),
    }
}

/// Emit the shared varyings interface struct: the `#[position]` field as
/// `@builtin(position)` (always first), then each varying at `@location(i)`
/// in field-declaration order. Integer varyings carry `@interpolate(flat)` —
/// WGSL requires flat interpolation on integer user IO, on BOTH the vertex
/// output and the fragment input (the WGSL twin of the SPIR-V `Flat`
/// decoration on both interface ends and the MSL `[[flat]]`); emitting the
/// struct once covers both, since it is byte-identical between the vertex
/// (output) and fragment (input) modules — the WGSL-native form of the
/// shared-struct model.
fn emit_varyings_struct(out: &mut String, v: &ShaderVaryings) {
    out.push_str(&format!("struct {} {{\n", v.struct_name));
    out.push_str(&format!(
        "    @builtin(position) {}: vec4<f32>,\n",
        v.position
    ));
    for (i, f) in v.fields.iter().enumerate() {
        let interp = if f.ty == ShaderType::U32 {
            " @interpolate(flat)"
        } else {
            ""
        };
        out.push_str(&format!(
            "    @location({}){interp} {}: {},\n",
            i,
            f.name,
            shader_type_wgsl(f.ty)
        ));
    }
    out.push_str("};\n\n");
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
    for (p, binding) in shader.params.iter().zip(bindings.iter()) {
        if p.is_slice {
            let binding = binding.expect("a slice param always has a shared binding index");
            out.push_str(&format!(
                "@group(0) @binding({}) var<storage, read> {}: array<{}>;\n",
                binding,
                p.name,
                shader_slice_elem_wgsl(p.ty),
            ));
        }
    }
}

/// Emit one `var<uniform>` per `&T` uniform param, at its shared decl-index
/// binding — `@group(0) @binding(slot) var<uniform> name: T;`. The `var<uniform>`
/// storage class (not `var<storage>`) matches the WebGPU driver, which allocates
/// these buffers with `FieldUsage::UNIFORM` (→ `buffer_usage::UNIFORM`), and the
/// compute-kernel `Constant` precedent (`emit_wgsl::kernel`). Slice bindings use
/// the same shared-index table so a uniform and a slice never collide.
fn emit_uniform_bindings(out: &mut String, shader: &ShaderDef, bindings: &[Option<u32>]) {
    for (p, binding) in shader.params.iter().zip(bindings.iter()) {
        if p.is_uniform {
            let binding = binding.expect("a uniform param always has a shared binding index");
            out.push_str(&format!(
                "@group(0) @binding({}) var<uniform> {}: {};\n",
                binding,
                p.name,
                shader_type_wgsl(p.ty),
            ));
        }
    }
}

/// The number of texture slots a body samples: `max(slot) + 1` over every
/// `sample(N, …)` in the body, or 0 if none. The scan is whitespace-tolerant
/// between `sample`, `(`, and the slot digit — a non-macro producer or a
/// printer change could space them apart — mirroring the MSL `body_samples_slot`
/// scan. Slots are assumed dense (0..max), the same shape both natives use.
fn texture_slot_count(body: &str) -> u32 {
    (0..8u32)
        .filter(|slot| body_samples_slot(body, *slot))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Whether `body` samples texture slot `slot`, tolerating whitespace between
/// `sample`, `(`, and the slot digit (`sample ( 0`, `sample( 0`, …). Byte-for-
/// byte the MSL emitter's scan so the two agree on which slots are bound.
fn body_samples_slot(body: &str, slot: u32) -> bool {
    let digit = char::from_digit(slot, 10).unwrap();
    let bytes = body.as_bytes();
    let mut i = 0;
    while let Some(rel) = body[i..].find("sample") {
        let mut j = i + rel + "sample".len();
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

/// Emit a `texture_2d<f32>` + `sampler` pair per sampled slot, at bindings
/// `8+slot` and past. WGSL requires the texture and its sampler as separate
/// bindings (unlike Metal's combined sampler); the DSL's `sample(N, uv)` lowers
/// to `textureSample(tex_N, smp_N, uv)` in the walker, so the names must be
/// `tex_N` / `smp_N`. The base of 8 keeps textures clear of the uniform/slice
/// binding space (which the cap holds to 0..8).
fn emit_texture_bindings(out: &mut String, tex_slots: u32) {
    for slot in 0..tex_slots {
        let tex_binding = TEXTURE_BINDING_BASE + slot * 2;
        let smp_binding = tex_binding + 1;
        out.push_str(&format!(
            "@group(0) @binding({tex_binding}) var tex_{slot}: texture_2d<f32>;\n"
        ));
        out.push_str(&format!(
            "@group(0) @binding({smp_binding}) var smp_{slot}: sampler;\n"
        ));
    }
}

/// Emit every module-scope binding (slices, uniforms, textures) followed by a
/// blank line when any were emitted. Shared by both stages so the ordering —
/// slices, then uniforms, then textures — is identical.
fn emit_module_bindings(
    out: &mut String,
    shader: &ShaderDef,
    bindings: &[Option<u32>],
    tex_slots: u32,
) {
    let before = out.len();
    emit_slice_storage_bindings(out, shader, bindings);
    emit_uniform_bindings(out, shader, bindings);
    emit_texture_bindings(out, tex_slots);
    if out.len() != before {
        out.push('\n');
    }
}

pub fn emit_vertex_shader(shader: &ShaderDef) -> Result<String, String> {
    let mut out = String::new();

    let bindings = shared_binding_indices(shader)?;
    let tex_slots = texture_slot_count(&shader.body_source);
    let infos: Vec<ParamInfo> = param_infos(&shader.params);

    // Attributes are the plain value params (neither uniform nor slice); a
    // slice binds a storage buffer, not a vertex attribute. Attributes are
    // PURE inputs — nothing is auto-forwarded; every varying is written
    // explicitly through the Varyings struct literal.
    let attr_params: Vec<&ShaderParam> = shader
        .params
        .iter()
        .filter(|p| !p.is_uniform && !p.is_slice)
        .collect();

    // Slice/uniform/texture bindings precede the interface structs.
    emit_module_bindings(&mut out, shader, &bindings, tex_slots);

    // The input struct exists only when there are attributes — WGSL rejects
    // an empty struct, and a builtin-only vertex (the `vertex_id()` unit-quad
    // synthesis, a fullscreen triangle) binds no vertex buffers at all. A u32
    // attribute is a plain `@location` u32 member (vertex inputs are fetched,
    // never interpolated — no `@interpolate` there, matching the undecorated
    // SPIR-V Input).
    if !attr_params.is_empty() {
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
    }

    // Entry-point params: the attribute struct, then the vertex-index
    // builtins — each declared only when the body calls it (whitespace-
    // tolerant scan, like the texture slots). The body walker lowers
    // `vertex_id()` / `instance_id()` to these exact identifiers.
    let mut fn_params: Vec<String> = Vec::new();
    if !attr_params.is_empty() {
        fn_params.push("in: VertexInput".to_string());
    }
    if body_calls(&shader.body_source, "vertex_id") {
        fn_params.push("@builtin(vertex_index) _vertex_id: u32".to_string());
    }
    if body_calls(&shader.body_source, "instance_id") {
        fn_params.push("@builtin(instance_index) _instance_id: u32".to_string());
    }
    let fn_params = fn_params.join(", ");

    // The entry function keeps the shader's REAL name — the runtime passes
    // `ShaderBinary.entry_point` (the `#[quanta::vertex]` fn name, which is
    // `shader.name` here) into `GPURenderPipelineDescriptor.entryPoint`, so a
    // `fn main` module fails every pipeline with "entry point doesn't exist".
    // MSL and SPIR-V already name their entries this way; the same exposure
    // to pathological names (a shader named after one of its own module-scope
    // bindings, or a WGSL reserved word) is accepted identically — it fails
    // loudly at validation, exactly as it would on the natives.
    if let Some(v) = &shader.varyings {
        // Shared-struct model: the out struct IS the varyings struct, and the
        // body's tail literal assigns every member explicitly.
        emit_varyings_struct(&mut out, v);

        out.push_str(&format!(
            "@vertex\nfn {}({fn_params}) -> {} {{\n",
            shader.name, v.struct_name
        ));
        for p in &attr_params {
            out.push_str(&format!("    let {} = in.{};\n", p.name, p.name));
        }
        out.push_str(&format!("    var _vout: {};\n", v.struct_name));
        walk_body_varyings(&shader.body_source, &infos, v, "_vout", "    ", &mut out)?;
        out.push_str("    return _vout;\n");
        out.push_str("}\n");
        return Ok(out);
    }

    // Position-only vertex (`-> Vec4`): no varyings at all.
    out.push_str("struct VertexOutput {\n");
    out.push_str("    @builtin(position) position: vec4<f32>,\n");
    out.push_str("};\n\n");

    out.push_str(&format!(
        "@vertex\nfn {}({fn_params}) -> VertexOutput {{\n",
        shader.name
    ));
    for p in &attr_params {
        out.push_str(&format!("    let {} = in.{};\n", p.name, p.name));
    }

    // Lower the body; the vertex tail is the clip-space position.
    let (pos_expr, _ty) = walk_body(
        &shader.body_source,
        &infos,
        None,
        crate::ShaderStage::Vertex,
        "    ",
        &mut out,
    )?;
    out.push_str("    var output: VertexOutput;\n");
    out.push_str(&format!("    output.position = {pos_expr};\n"));
    out.push_str("    return output;\n");
    out.push_str("}\n");

    Ok(out)
}

pub fn emit_fragment_shader(shader: &ShaderDef) -> Result<String, String> {
    reject_fragment_value_params(shader)?;
    let mut out = String::new();

    let bindings = shared_binding_indices(shader)?;
    let tex_slots = texture_slot_count(&shader.body_source);
    let infos: Vec<ParamInfo> = param_infos(&shader.params);

    // Slice/uniform/texture bindings precede the interface struct.
    emit_module_bindings(&mut out, shader, &bindings, tex_slots);

    if let Some(v) = &shader.varyings {
        // Shared-struct model: the fragment takes the varyings struct as its
        // single stage input, named by the receiver param; the body reads
        // fields as `<receiver>.<field>` (the position member is the
        // interpolated window position — WGSL FragCoord semantics, and the
        // member `frag_coord()` reads through, so no extra builtin param is
        // declared here).
        emit_varyings_struct(&mut out, v);
        let recv = v.binding.as_deref().ok_or_else(|| {
            format!(
                "fragment shader `{}`: the varyings interface names no receiver param",
                shader.name
            )
        })?;
        // Real entry name — same contract as the vertex emitter (see the
        // comment there): the runtime's `fragment_entry` names this.
        out.push_str(&format!(
            "@fragment\nfn {}({recv}: {}) -> @location(0) vec4<f32> {{\n",
            shader.name, v.struct_name
        ));
        let (color_expr, _ty) = walk_body(
            &shader.body_source,
            &infos,
            Some(v),
            crate::ShaderStage::Fragment,
            "    ",
            &mut out,
        )?;
        out.push_str(&format!("    return {color_expr};\n"));
        out.push_str("}\n");
        return Ok(out);
    }

    // No varyings: the fragment reads only uniforms/slices/textures — plus
    // the window position when the body calls `frag_coord()` (declared only
    // then, like the vertex-index builtins; the walker lowers the call to
    // this exact identifier).
    if body_calls(&shader.body_source, "frag_coord") {
        out.push_str(&format!(
            "@fragment\nfn {}(@builtin(position) _frag_coord: vec4<f32>) \
             -> @location(0) vec4<f32> {{\n",
            shader.name
        ));
    } else {
        out.push_str(&format!(
            "@fragment\nfn {}() -> @location(0) vec4<f32> {{\n",
            shader.name
        ));
    }

    // Lower the body; the fragment tail is the output color.
    let (color_expr, _ty) = walk_body(
        &shader.body_source,
        &infos,
        None,
        crate::ShaderStage::Fragment,
        "    ",
        &mut out,
    )?;
    out.push_str(&format!("    return {color_expr};\n"));
    out.push_str("}\n");

    Ok(out)
}
