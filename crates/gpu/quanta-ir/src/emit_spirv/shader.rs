//! Vertex and fragment shader SPIR-V emission.
//!
//! Generates passthrough vertex/fragment shaders: vertex attributes become
//! Input variables, the first is promoted to gl_Position (vertex) or
//! output color (fragment).

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    // ── Vertex shader ───────────────────────────────────────────────────────

    /// Emit a vertex shader SPIR-V module.
    ///
    /// Generates a passthrough vertex shader: each value parameter becomes
    /// an Input variable with Location decoration. The first parameter is
    /// promoted to gl_Position output (expanded to vec4 with w=1.0).
    /// Uniform parameters are ignored for V1 (no push constant block yet).
    pub(crate) fn emit_vertex_shader(&mut self, shader: &crate::ShaderDef) -> Result<(), String> {
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
        let attr_params: Vec<(usize, &crate::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_uniform)
            .collect();

        let mut interface_ids = Vec::new();
        let mut input_vars: Vec<(u32, u32)> = Vec::new(); // (var_id, type_id)

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

        // 3. Declare Output variable: gl_Position (BuiltIn Position, vec4)
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

        // No execution mode for vertex shaders (no LocalSize, no OriginUpperLeft)

        // 5. Function body
        Self::emit_op(
            &mut self.sec_function,
            OP_FUNCTION,
            &[void_ty, main_id, FUNCTION_CONTROL_NONE, func_ty],
        );
        let entry_label = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[entry_label]);

        // Load the first input and construct vec4 for gl_Position.
        // If no inputs, emit a zero position.
        let result_id = self.passthrough_first_input(&attr_params, &input_vars, f32_ty, vec4_ty)?;

        // Store to gl_Position
        Self::emit_op(&mut self.sec_function, OP_STORE, &[position_var, result_id]);

        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(())
    }

    // ── Fragment shader ─────────────────────────────────────────────────────

    /// Emit a fragment shader SPIR-V module.
    ///
    /// Generates a passthrough fragment shader: each value parameter becomes
    /// an Input variable (interpolated varying) with Location decoration.
    /// The first input is passed through to Location(0) Output.
    pub(crate) fn emit_fragment_shader(&mut self, shader: &crate::ShaderDef) -> Result<(), String> {
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
        let stage_in_params: Vec<(usize, &crate::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_uniform)
            .collect();

        let mut interface_ids = Vec::new();
        let mut input_vars: Vec<(u32, u32)> = Vec::new(); // (var_id, type_id)

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

        // 3. Declare Output variable: fragment color at Location(0)
        let f32_ty = self.ensure_type_f32();
        let vec4_ty = self.ensure_type_vector(f32_ty, 4);
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

        // 5. Execution mode: OriginUpperLeft (required for Fragment)
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

        // Load the first input and promote to vec4 for the output color.
        let result_id =
            self.passthrough_first_input(&stage_in_params, &input_vars, f32_ty, vec4_ty)?;

        // Store to output color
        Self::emit_op(&mut self.sec_function, OP_STORE, &[color_var, result_id]);

        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(())
    }

    // ── Shared helper ───────────────────────────────────────────────────────

    /// Load first input, promote to vec4. Shared by vertex and fragment shaders.
    pub(crate) fn passthrough_first_input(
        &mut self,
        attr_params: &[(usize, &crate::ShaderParam)],
        input_vars: &[(u32, u32)],
        f32_ty: u32,
        vec4_ty: u32,
    ) -> Result<u32, String> {
        if input_vars.is_empty() {
            // No inputs — emit vec4(0,0,0,1) for vertex or vec4(1,1,1,1) for fragment
            let zero = self.emit_constant_f32(0.0);
            let one = self.emit_constant_f32(1.0);
            let result = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_COMPOSITE_CONSTRUCT,
                &[vec4_ty, result, zero, zero, zero, one],
            );
            return Ok(result);
        }

        let (first_var, first_ty) = input_vars[0];
        let loaded = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_LOAD,
            &[first_ty, loaded, first_var],
        );

        // Promote to vec4 based on component count
        let components = Self::shader_type_components(attr_params[0].1.ty);
        match components {
            4 => Ok(loaded), // Already vec4
            3 => {
                // vec3 -> vec4(pos.x, pos.y, pos.z, 1.0)
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
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, result, x, y, z, one],
                );
                Ok(result)
            }
            2 => {
                // vec2 -> vec4(pos.x, pos.y, 0.0, 1.0)
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
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, result, x, y, zero, one],
                );
                Ok(result)
            }
            1 => {
                // f32 -> vec4(val, 0.0, 0.0, 1.0)
                let zero = self.emit_constant_f32(0.0);
                let one = self.emit_constant_f32(1.0);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, result, loaded, zero, zero, one],
                );
                Ok(result)
            }
            _ => Err(format!(
                "unsupported component count {} for shader input",
                components
            )),
        }
    }
}
