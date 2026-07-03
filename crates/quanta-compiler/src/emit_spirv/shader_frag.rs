//! Fragment shader SPIR-V emission.
//!
//! Generates a fragment shader: each value parameter becomes an Input
//! variable (interpolated varying) with Location decoration. The body
//! expression is evaluated and written to Location(0) output.

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    /// Emit a fragment shader SPIR-V module.
    pub(crate) fn emit_fragment_shader(
        &mut self,
        shader: &quanta_ir::ShaderDef,
    ) -> Result<(), String> {
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
            interface_ids.push(var_id);
            input_vars.push((var_id, ty_id));
        }

        // 2b. Declare combined image samplers for texture sampling
        let f32_ty = self.ensure_type_f32();
        let vec4_ty = self.ensure_type_vector(f32_ty, 4);

        let max_tex_slot = (0..8u32)
            .filter(|slot| shader.body_source.contains(&format!("sample({}", slot)))
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

        let param_info: Vec<(String, u32, u32, quanta_ir::ShaderType)> = stage_in_params
            .iter()
            .zip(input_vars.iter())
            .map(|((_, p), (var_id, type_id))| (p.name.clone(), *var_id, *type_id, p.ty))
            .collect();

        let saved_func = self.sec_function.clone();
        let saved_next = self.next_id;

        let result_id = match self.eval_shader_body(&shader.body_source, &param_info) {
            Ok((id, ty)) => self
                .promote_to_vec4(id, ty, f32_ty, vec4_ty)
                .unwrap_or_else(|| {
                    self.sec_function = saved_func.clone();
                    self.next_id = saved_next;
                    self.passthrough_first_input(&stage_in_params, &input_vars, f32_ty, vec4_ty)
                }),
            Err(_) => {
                self.sec_function = saved_func;
                self.next_id = saved_next;
                self.passthrough_first_input(&stage_in_params, &input_vars, f32_ty, vec4_ty)
            }
        };

        Self::emit_op(&mut self.sec_function, OP_STORE, &[color_var, result_id]);
        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(())
    }
}
