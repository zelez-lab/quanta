//! Vertex/fragment shader WGSL emitters.
//!
//! The interface shell (input/output structs, uniform + slice + texture
//! bindings) is emitted here; the function BODY is lowered by the hand-rolled
//! recursive-descent walker in [`super::shader_walker`], which re-tokenizes the
//! token-stringified Rust body and emits real WGSL statements — so `let`/`let
//! mut`, statement-`if`/`else`, value-`if`, `&T` uniform derefs, `&[T]` slice
//! indexing, swizzles, intrinsics, and `sample(N, uv)` all translate. The
//! construct surface mirrors the SPIR-V shader walker: what SPIR-V accepts,
//! this accepts; what SPIR-V rejects, this rejects with a clear error.

use super::shader_walker::{ParamInfo, param_infos, walk_body};
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
        // Interface spelling only; u32 params are rejected before use — see
        // `reject_u32_params` (varyings would need `@interpolate(flat)` and the
        // walker would need u32-typed literals/comparisons).
        ShaderType::U32 => "u32",
    }
}

/// The WGSL emitter does not support `u32` shader params yet: a u32 varying
/// needs `@interpolate(flat)` on both interface structs, and the body walker
/// would emit float-typed literals against it (WGSL has no implicit
/// conversions, so `naga` rejects the module). Fail emission with a named gap
/// — the shader ships with `wgsl: None` and a build-time note, like the other
/// documented WGSL gaps — instead of emitting invalid WGSL.
fn reject_u32_params(shader: &ShaderDef) -> Result<(), String> {
    match shader.params.iter().find(|p| p.ty == ShaderType::U32) {
        Some(p) => Err(format!(
            "shader `{}` param `{}`: u32 shader params are not yet supported by \
             the WGSL emitter",
            shader.name, p.name
        )),
        None => Ok(()),
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
    reject_u32_params(shader)?;
    let mut out = String::new();

    let bindings = shared_binding_indices(shader)?;
    let tex_slots = texture_slot_count(&shader.body_source);
    let infos: Vec<ParamInfo> = param_infos(&shader.params);

    // Attributes are the plain value params (neither uniform nor slice); a
    // slice binds a storage buffer, not a vertex attribute.
    let attr_params: Vec<&ShaderParam> = shader
        .params
        .iter()
        .filter(|p| !p.is_uniform && !p.is_slice)
        .collect();
    let varying_params: Vec<&ShaderParam> = attr_params.iter().skip(1).copied().collect();

    // Slice/uniform/texture bindings precede the interface structs.
    emit_module_bindings(&mut out, shader, &bindings, tex_slots);

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

    // Lower the body; the vertex tail is the clip-space position.
    let (pos_expr, _ty) = walk_body(&shader.body_source, &infos, "    ", &mut out)?;
    out.push_str("    var output: VertexOutput;\n");
    out.push_str(&format!("    output.position = {pos_expr};\n"));
    for p in &varying_params {
        out.push_str(&format!("    output.{} = {};\n", p.name, p.name));
    }
    out.push_str("    return output;\n");
    out.push_str("}\n");

    Ok(out)
}

pub fn emit_fragment_shader(shader: &ShaderDef) -> Result<String, String> {
    reject_u32_params(shader)?;
    let mut out = String::new();

    let bindings = shared_binding_indices(shader)?;
    let tex_slots = texture_slot_count(&shader.body_source);
    let infos: Vec<ParamInfo> = param_infos(&shader.params);

    // Interpolated inputs are the plain value params (neither uniform nor
    // slice); a slice binds a storage buffer, not a stage input.
    let stage_in_params: Vec<&ShaderParam> = shader
        .params
        .iter()
        .filter(|p| !p.is_uniform && !p.is_slice)
        .collect();

    // Slice/uniform/texture bindings precede the interface struct.
    emit_module_bindings(&mut out, shader, &bindings, tex_slots);

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

    // Lower the body; the fragment tail is the output color.
    let (color_expr, _ty) = walk_body(&shader.body_source, &infos, "    ", &mut out)?;
    out.push_str(&format!("    return {color_expr};\n"));
    out.push_str("}\n");

    Ok(out)
}
