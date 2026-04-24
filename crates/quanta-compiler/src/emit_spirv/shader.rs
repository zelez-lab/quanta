//! Vertex shader SPIR-V emission.
//!
//! Emits a vertex shader module with Input variables (vertex attributes),
//! Output variables (varyings + gl_Position), push constant uniforms,
//! and the body_source expression evaluated as gl_Position.

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    /// Pass-through fallback: load first input, promote to vec4.
    /// Used when body evaluation fails (unsupported features like uniforms).
    pub(crate) fn passthrough_first_input(
        &mut self,
        attr_params: &[(usize, &quanta_ir::ShaderParam)],
        input_vars: &[(u32, u32)],
        f32_ty: u32,
        vec4_ty: u32,
    ) -> u32 {
        if input_vars.is_empty() {
            let zero = self.emit_constant_f32(0.0);
            let one = self.emit_constant_f32(1.0);
            let r = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_COMPOSITE_CONSTRUCT,
                &[vec4_ty, r, zero, zero, zero, one],
            );
            return r;
        }
        let (var_id, type_id) = input_vars[0];
        let loaded = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LOAD, &[type_id, loaded, var_id]);
        let comps = Self::shader_type_components(attr_params[0].1.ty);
        match comps {
            4 => loaded,
            3 => {
                let x = self.alloc_id();
                let y = self.alloc_id();
                let z = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, x, loaded, 0],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, y, loaded, 1],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, z, loaded, 2],
                );
                let one = self.emit_constant_f32(1.0);
                let r = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, r, x, y, z, one],
                );
                r
            }
            2 => {
                let x = self.alloc_id();
                let y = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, x, loaded, 0],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, y, loaded, 1],
                );
                let zero = self.emit_constant_f32(0.0);
                let one = self.emit_constant_f32(1.0);
                let r = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, r, x, y, zero, one],
                );
                r
            }
            _ => {
                let zero = self.emit_constant_f32(0.0);
                let one = self.emit_constant_f32(1.0);
                let r = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, r, loaded, zero, zero, one],
                );
                r
            }
        }
    }

    /// Promote a shader result to vec4 (for gl_Position or fragment output).
    pub(crate) fn promote_to_vec4(
        &mut self,
        id: u32,
        ty: quanta_ir::ShaderType,
        f32_ty: u32,
        vec4_ty: u32,
    ) -> Option<u32> {
        match ty {
            quanta_ir::ShaderType::Vec4 => Some(id),
            quanta_ir::ShaderType::Vec3 => {
                let x = self.alloc_id();
                let y = self.alloc_id();
                let z = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, x, id, 0],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, y, id, 1],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, z, id, 2],
                );
                let one = self.emit_constant_f32(1.0);
                let r = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, r, x, y, z, one],
                );
                Some(r)
            }
            quanta_ir::ShaderType::Vec2 => {
                let x = self.alloc_id();
                let y = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, x, id, 0],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, y, id, 1],
                );
                let zero = self.emit_constant_f32(0.0);
                let one = self.emit_constant_f32(1.0);
                let r = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, r, x, y, zero, one],
                );
                Some(r)
            }
            quanta_ir::ShaderType::F32 => {
                let zero = self.emit_constant_f32(0.0);
                let one = self.emit_constant_f32(1.0);
                let r = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, r, id, zero, zero, one],
                );
                Some(r)
            }
            _ => None,
        }
    }

    /// Emit a vertex shader SPIR-V module.
    pub(crate) fn emit_vertex_shader(
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
        let attr_params: Vec<(usize, &quanta_ir::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_uniform)
            .collect();

        let mut interface_ids = Vec::new();
        let mut input_vars: Vec<(u32, u32)> = Vec::new();

        for (loc, (_, param)) in attr_params.iter().enumerate() {
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

        // 2b. Declare uniform params as push constant struct
        let uniform_params: Vec<(usize, &quanta_ir::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_uniform)
            .collect();

        let mut uniform_vars: Vec<(String, u32, u32, quanta_ir::ShaderType)> = Vec::new();
        if !uniform_params.is_empty() {
            self.emit_uniform_push_constants(&uniform_params, &mut uniform_vars);
        }

        // 2c. Declare Output variables for varyings
        let mut varying_outputs: Vec<(u32, u32, u32)> = Vec::new();
        for (i, (_, param)) in attr_params.iter().enumerate().skip(1) {
            let varying_loc = (i - 1) as u32;
            let ty_id = self.shader_type_id(param.ty);
            let ptr_ty = self.ensure_type_pointer(STORAGE_CLASS_OUTPUT, ty_id);
            let out_var = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_ty, out_var, STORAGE_CLASS_OUTPUT],
            );
            self.emit_name(out_var, &format!("out_{}", param.name));
            self.decorate(out_var, DECORATION_LOCATION, &[varying_loc]);
            interface_ids.push(out_var);
            varying_outputs.push((out_var, ty_id, input_vars[i].0));
        }

        // 3. Declare gl_Position
        let f32_ty = self.ensure_type_f32();
        let vec4_ty = self.ensure_type_vector(f32_ty, 4);
        let ptr_output_vec4 = self.ensure_type_pointer(STORAGE_CLASS_OUTPUT, vec4_ty);
        let position_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_output_vec4, position_var, STORAGE_CLASS_OUTPUT],
        );
        self.emit_name(position_var, "gl_Position");
        self.decorate(position_var, DECORATION_BUILTIN, &[BUILTIN_POSITION]);
        interface_ids.push(position_var);

        // 4. Entry point
        let void_ty = self.ensure_type_void();
        let func_ty = self.ensure_type_function(void_ty, &[]);
        let main_id = self.alloc_id();
        self.emit_name(main_id, "main");
        {
            let name_words = Self::string_words("main");
            let mut ops = vec![EXECUTION_MODEL_VERTEX, main_id];
            ops.extend_from_slice(&name_words);
            ops.extend_from_slice(&interface_ids);
            Self::emit_op(&mut self.sec_entry_point, OP_ENTRY_POINT, &ops);
        }

        // 5. Function body
        Self::emit_op(
            &mut self.sec_function,
            OP_FUNCTION,
            &[void_ty, main_id, FUNCTION_CONTROL_NONE, func_ty],
        );
        let entry_label = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[entry_label]);

        // Build param_info
        let mut param_info: Vec<(String, u32, u32, quanta_ir::ShaderType)> = attr_params
            .iter()
            .zip(input_vars.iter())
            .map(|((_, p), (var_id, type_id))| (p.name.clone(), *var_id, *type_id, p.ty))
            .collect();

        // Emit AccessChain for uniforms
        for (member_idx, (name, pc_var, member_ty, sty)) in uniform_vars.iter().enumerate() {
            let ptr_member = self.ensure_type_pointer(STORAGE_CLASS_PUSH_CONSTANT, *member_ty);
            let idx_const = self.emit_constant_u32(member_idx as u32);
            let access = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_ACCESS_CHAIN,
                &[ptr_member, access, *pc_var, idx_const],
            );
            param_info.push((name.clone(), access, *member_ty, *sty));
        }

        // Forward vertex attributes as varying outputs
        for (out_var, type_id, in_var) in &varying_outputs {
            let loaded = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_LOAD,
                &[*type_id, loaded, *in_var],
            );
            Self::emit_op(&mut self.sec_function, OP_STORE, &[*out_var, loaded]);
        }

        // Evaluate body or fall back to pass-through
        let saved_func = self.sec_function.clone();
        let saved_next = self.next_id;

        let result_id = match self.eval_shader_body(&shader.body_source, &param_info) {
            Ok((id, ty)) => self
                .promote_to_vec4(id, ty, f32_ty, vec4_ty)
                .unwrap_or_else(|| {
                    self.sec_function = saved_func.clone();
                    self.next_id = saved_next;
                    self.passthrough_first_input(&attr_params, &input_vars, f32_ty, vec4_ty)
                }),
            Err(_) => {
                self.sec_function = saved_func;
                self.next_id = saved_next;
                self.passthrough_first_input(&attr_params, &input_vars, f32_ty, vec4_ty)
            }
        };

        Self::emit_op(&mut self.sec_function, OP_STORE, &[position_var, result_id]);
        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(())
    }

    /// Emit uniform push constant struct for shader uniforms.
    pub(crate) fn emit_uniform_push_constants(
        &mut self,
        uniform_params: &[(usize, &quanta_ir::ShaderParam)],
        uniform_vars: &mut Vec<(String, u32, u32, quanta_ir::ShaderType)>,
    ) {
        let mut member_types = Vec::new();
        let mut member_offsets = Vec::new();
        let mut offset = 0u32;
        for (_, p) in uniform_params {
            let ty_id = self.shader_type_id(p.ty);
            member_types.push(ty_id);
            member_offsets.push(offset);
            let size = match p.ty {
                quanta_ir::ShaderType::Mat4 => 64u32,
                quanta_ir::ShaderType::Mat3 => 48,
                quanta_ir::ShaderType::Vec4 => 16,
                quanta_ir::ShaderType::Vec3 => 16,
                quanta_ir::ShaderType::Vec2 => 8,
                quanta_ir::ShaderType::F32 => 4,
            };
            offset += size;
        }

        let struct_ty = self.alloc_id();
        let mut struct_ops = vec![struct_ty];
        struct_ops.extend_from_slice(&member_types);
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_STRUCT, &struct_ops);
        self.decorate(struct_ty, DECORATION_BLOCK, &[]);
        for (i, off) in member_offsets.iter().enumerate() {
            self.member_decorate(struct_ty, i as u32, DECORATION_OFFSET, &[*off]);
            if matches!(
                uniform_params[i].1.ty,
                quanta_ir::ShaderType::Mat4 | quanta_ir::ShaderType::Mat3
            ) {
                self.member_decorate(struct_ty, i as u32, 5 /* ColMajor */, &[]);
                let stride = match uniform_params[i].1.ty {
                    quanta_ir::ShaderType::Mat4 => 16u32,
                    quanta_ir::ShaderType::Mat3 => 16,
                    _ => 16,
                };
                self.member_decorate(struct_ty, i as u32, 7 /* MatrixStride */, &[stride]);
            }
        }

        let ptr_pc = self.ensure_type_pointer(STORAGE_CLASS_PUSH_CONSTANT, struct_ty);
        let pc_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_pc, pc_var, STORAGE_CLASS_PUSH_CONSTANT],
        );
        self.emit_name(pc_var, "push_constants");

        for (_, p) in uniform_params {
            let mty = self.shader_type_id(p.ty);
            uniform_vars.push((p.name.clone(), pc_var, mty, p.ty));
        }
    }
}
