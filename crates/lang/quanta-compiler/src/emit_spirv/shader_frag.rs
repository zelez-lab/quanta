//! Fragment shader SPIR-V emission.
//!
//! Generates a fragment shader: each value parameter becomes an Input
//! variable (interpolated varying) with Location decoration. The body
//! expression is evaluated and written to Location(0) output.

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    /// Emit a fragment shader SPIR-V module.
    ///
    /// With `passthrough == false` the body is translated; a body the
    /// translator can't handle logs a warning and returns
    /// [`ShaderEmit::NeedsPassthrough`] so the caller can rebuild on a fresh
    /// emitter. With `passthrough == true` the body is skipped and the
    /// interface-only passthrough (load input[0] → colour) is emitted directly
    /// — the two calls run identical interface setup, so the fresh passthrough
    /// module is id-consistent by construction.
    pub(crate) fn emit_fragment_shader(
        &mut self,
        shader: &quanta_ir::ShaderDef,
        passthrough: bool,
    ) -> Result<super::ShaderEmit, String> {
        // 1. Capability + memory model
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_SHADER],
        );
        Self::emit_op(
            &mut self.sec_memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        // 2. Declare Input variables for value params
        let stage_in_params: Vec<(usize, &quanta_ir::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_uniform)
            .collect();

        let mut interface_ids = Vec::new();
        let mut input_vars: Vec<(u32, u32)> = Vec::new();

        for (loc, (_, param)) in stage_in_params.iter().enumerate() {
            let ty_id = self.shader_type_id(param.ty);
            let ptr_ty = self.ensure_type_pointer(STORAGE_CLASS_INPUT, ty_id);
            let var_id = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_ty, var_id, STORAGE_CLASS_INPUT],
            );
            self.emit_name(var_id, &param.name);
            self.decorate(var_id, DECORATION_LOCATION, &[loc as u32]);
            // An integer fragment Input MUST be Flat — integers cannot be
            // interpolated, and spirv-val/drivers reject a non-Flat int
            // interpolant (VUID-StandaloneSpirv-Flat-04744). Matches the Flat
            // on the vertex emitter's u32 varying Output. Float inputs stay
            // smooth (undecorated).
            if param.ty == quanta_ir::ShaderType::U32 {
                self.decorate(var_id, DECORATION_FLAT, &[]);
            }
            interface_ids.push(var_id);
            input_vars.push((var_id, ty_id));
        }

        // 2b. Declare combined image samplers for texture sampling
        let f32_ty = self.ensure_type_f32();
        let vec4_ty = self.ensure_type_vector(f32_ty, 4);

        let max_tex_slot = (0..8u32)
            .filter(|slot| body_samples_slot(&shader.body_source, *slot))
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        self.texture_samplers.clear();
        self.texture_image_types.clear();
        if max_tex_slot > 0 {
            let image_ty = self.alloc_id();
            Self::emit_op(
                &mut self.sec_type_const,
                OP_TYPE_IMAGE,
                &[image_ty, f32_ty, 1, 0, 0, 0, 1, 0],
            );
            let sampled_image_ty = self.alloc_id();
            Self::emit_op(
                &mut self.sec_type_const,
                OP_TYPE_SAMPLED_IMAGE,
                &[sampled_image_ty, image_ty],
            );
            let ptr_uniform_si =
                self.ensure_type_pointer(STORAGE_CLASS_UNIFORM_CONSTANT, sampled_image_ty);

            for slot in 0..max_tex_slot {
                let var_id = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_global_var,
                    OP_VARIABLE,
                    &[ptr_uniform_si, var_id, STORAGE_CLASS_UNIFORM_CONSTANT],
                );
                self.emit_name(var_id, &format!("tex_{}", slot));
                self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
                self.decorate(var_id, DECORATION_BINDING, &[slot + 8]);
                self.texture_samplers
                    .insert(slot, (var_id, sampled_image_ty));
            }
        }

        // 2b'. FragCoord builtin: an Input vec4 decorated `BuiltIn FragCoord`
        // (the Input analogue of the vertex emitter's gl_Position Output),
        // declared only when the body calls `frag_coord()` — an unused builtin
        // input just bloats the interface. The scan is deterministic over the
        // same body, so the real and passthrough calls declare it identically
        // (the id-consistency contract in the doc comment above).
        self.frag_coord_var = None;
        if body_uses_frag_coord(&shader.body_source) {
            let ptr_input_vec4 = self.ensure_type_pointer(STORAGE_CLASS_INPUT, vec4_ty);
            let var_id = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_input_vec4, var_id, STORAGE_CLASS_INPUT],
            );
            self.emit_name(var_id, "gl_FragCoord");
            self.decorate(var_id, DECORATION_BUILTIN, &[BUILTIN_FRAG_COORD]);
            interface_ids.push(var_id);
            self.frag_coord_var = Some(var_id);
        }

        // 2c. Fragment uniforms + slices — shared storage-block emission over
        // one decl-index binding space (see emit_uniform_storage_blocks /
        // emit_slice_storage_blocks); the combined-cap error surfaces here.
        let bindings = super::shared_binding_indices(shader)?;
        let uniform_params: Vec<(usize, &quanta_ir::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_uniform)
            .collect();
        let slice_params: Vec<(usize, &quanta_ir::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_slice)
            .collect();

        self.slice_params.clear();
        let mut uniform_vars: Vec<(String, u32, u32, quanta_ir::ShaderType)> = Vec::new();
        if !uniform_params.is_empty() {
            self.emit_uniform_storage_blocks(
                &uniform_params,
                &bindings.uniform_bindings,
                &mut uniform_vars,
            );
        }
        if !slice_params.is_empty() {
            self.emit_slice_storage_blocks(&slice_params, &bindings.slice_bindings);
        }

        // 3. Declare Output variable: fragment color at Location(0)
        let ptr_output_vec4 = self.ensure_type_pointer(STORAGE_CLASS_OUTPUT, vec4_ty);
        let color_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_output_vec4, color_var, STORAGE_CLASS_OUTPUT],
        );
        self.emit_name(color_var, "out_color");
        self.decorate(color_var, DECORATION_LOCATION, &[0]);
        interface_ids.push(color_var);

        // 4. Entry point
        let void_ty = self.ensure_type_void();
        let func_ty = self.ensure_type_function(void_ty, &[]);
        let main_id = self.alloc_id();
        self.emit_name(main_id, "main");
        {
            let name_words = Self::string_words("main");
            let mut ops = vec![EXECUTION_MODEL_FRAGMENT, main_id];
            ops.extend_from_slice(&name_words);
            ops.extend_from_slice(&interface_ids);
            Self::emit_op(&mut self.sec_entry_point, OP_ENTRY_POINT, &ops);
        }

        // 5. Execution mode: OriginUpperLeft
        Self::emit_op(
            &mut self.sec_execution_mode,
            OP_EXECUTION_MODE,
            &[main_id, EXECUTION_MODE_ORIGIN_UPPER_LEFT],
        );

        // 6. Function body
        Self::emit_op(
            &mut self.sec_function,
            OP_FUNCTION,
            &[void_ty, main_id, FUNCTION_CONTROL_NONE, func_ty],
        );
        let entry_label = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[entry_label]);
        self.current_block = entry_label;

        let mut param_info: Vec<(String, u32, u32, quanta_ir::ShaderType)> = stage_in_params
            .iter()
            .zip(input_vars.iter())
            .map(|((_, p), (var_id, type_id))| (p.name.clone(), *var_id, *type_id, p.ty))
            .collect();

        // Uniforms: pointer to member 0 of each block; the expression
        // parser loads through it at use, exactly like an Input var.
        for (name, var_id, member_ty, sty) in &uniform_vars {
            let ptr_member = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, *member_ty);
            let zero = self.emit_constant_u32(0);
            let access = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_ACCESS_CHAIN,
                &[ptr_member, access, *var_id, zero],
            );
            param_info.push((name.clone(), access, *member_ty, *sty));
        }

        // Passthrough rebuild: skip the body, emit the interface-only result.
        if passthrough {
            let result_id =
                self.passthrough_first_input(&stage_in_params, &input_vars, f32_ty, vec4_ty);
            Self::emit_op(&mut self.sec_function, OP_STORE, &[color_var, result_id]);
            Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
            Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);
            return Ok(super::ShaderEmit::Real);
        }

        // Real attempt: translate the body. A failure interns ids into other
        // sections, so we abandon this emitter and let the caller rebuild the
        // passthrough on a fresh one rather than patching the id state here.
        let result_id = match self.eval_shader_body(&shader.body_source, &param_info) {
            Ok((id, ty)) => match self.promote_to_vec4(id, ty, f32_ty, vec4_ty) {
                Some(id) => id,
                None => {
                    eprintln!(
                        "[quanta] warning: fragment shader `{}` body result could not be promoted to Vec4; \
                     emitting a passthrough SPIR-V shader — it will MISRENDER \
                     on Vulkan (Metal/metallib is unaffected)",
                        shader.name
                    );
                    return Ok(super::ShaderEmit::NeedsPassthrough);
                }
            },
            Err(e) => {
                eprintln!(
                    "[quanta] warning: fragment shader `{}` body failed SPIR-V translation ({e}); \
                     emitting a passthrough SPIR-V shader — it will MISRENDER \
                     on Vulkan (Metal/metallib is unaffected)",
                    shader.name
                );
                return Ok(super::ShaderEmit::NeedsPassthrough);
            }
        };

        Self::emit_op(&mut self.sec_function, OP_STORE, &[color_var, result_id]);
        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(super::ShaderEmit::Real)
    }
}

/// Whether `body` samples texture slot `slot`, tolerating whitespace between
/// `sample`, `(`, and the slot digit (`sample ( 0`, `sample( 0`, …). Real macro
/// output keeps `sample(N` contiguous, but any other ShaderDef producer (or a
/// future printer change) could space them apart, so the slot scan must not
/// depend on a contiguous form.
/// Whether `body` calls the `frag_coord()` builtin, tolerating whitespace
/// between `frag_coord` and `(` — the same scan contract as
/// [`body_samples_slot`]. Only the call form counts: the DSL has no
/// user-defined functions, so an identifier followed by `(` can only be a
/// builtin call, and a param or local whose NAME contains the substring is
/// never followed by `(` and does not trigger a declaration.
fn body_uses_frag_coord(body: &str) -> bool {
    let bytes = body.as_bytes();
    let mut i = 0;
    while let Some(rel) = body[i..].find("frag_coord") {
        let mut j = i + rel + "frag_coord".len();
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j < bytes.len() && bytes[j] == b'(' {
            return true;
        }
        i += rel + "frag_coord".len();
    }
    false
}

fn body_samples_slot(body: &str, slot: u32) -> bool {
    let digit = char::from_digit(slot, 10).unwrap() as u8;
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
            if j < bytes.len() && bytes[j] == digit {
                return true;
            }
        }
        i += rel + "sample".len();
    }
    false
}
