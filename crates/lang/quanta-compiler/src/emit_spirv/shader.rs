//! Vertex shader SPIR-V emission.
//!
//! Emits a vertex shader module with Input variables (vertex attributes),
//! Output variables (gl_Position + the explicit varyings), and uniform /
//! slice storage blocks.
//!
//! Varyings follow the shared-struct model (`ShaderDef::varyings`): the body
//! ends in the Varyings struct literal, and each field is stored to its named
//! Output at the Location given by field-declaration order (u32 fields Flat).
//! Nothing is auto-forwarded — a vertex without a varyings interface writes
//! only gl_Position (its value params are pure attributes).

use std::collections::HashMap;

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
        let mut loaded = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LOAD, &[type_id, loaded, var_id]);
        // A u32 first input widens to f32 before the vec4 promotion below —
        // an OpCompositeConstruct with a uint member in a float vec4 is
        // invalid SPIR-V, even in a passthrough module.
        if attr_params[0].1.ty == quanta_ir::ShaderType::U32 {
            let widened = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_CONVERT_U_TO_F,
                &[f32_ty, widened, loaded],
            );
            loaded = widened;
        }
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
            // A u32 result widens to f32 (value-preserving), then promotes
            // like a scalar — a uint member in a float vec4 is invalid.
            quanta_ir::ShaderType::U32 => {
                let widened = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_CONVERT_U_TO_F,
                    &[f32_ty, widened, id],
                );
                self.promote_to_vec4(widened, quanta_ir::ShaderType::F32, f32_ty, vec4_ty)
            }
            _ => None,
        }
    }

    /// Emit a vertex shader SPIR-V module.
    ///
    /// With `passthrough == false` the body is translated; a body the
    /// translator can't handle logs a warning and returns
    /// [`ShaderEmit::NeedsPassthrough`] so the caller can rebuild on a fresh
    /// emitter. With `passthrough == true` the body is skipped and the
    /// interface-only passthrough (load input[0] → gl_Position) is emitted
    /// directly — the two calls run identical interface setup, so the fresh
    /// passthrough module is id-consistent by construction.
    pub(crate) fn emit_vertex_shader(
        &mut self,
        shader: &quanta_ir::ShaderDef,
        passthrough: bool,
    ) -> Result<super::ShaderEmit, String> {
        // FragCoord is a fragment-only builtin: never inherit one here, so a
        // vertex body calling `frag_coord()` errors in the body parser (and
        // falls to the passthrough) instead of loading a stale id. The
        // fragment-side varying-input state is likewise reset — the vertex
        // WRITES varyings (through the tail struct literal), never reads them.
        self.frag_coord_var = None;
        self.varyings = None;
        self.varying_inputs.clear();

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

        // 2. Declare Input variables for value params — the vertex
        // attributes, Location by declaration order. Pure inputs: nothing is
        // forwarded anywhere implicitly. (Slices bind storage buffers, not
        // attributes, so they are excluded alongside uniforms.)
        let attr_params: Vec<(usize, &quanta_ir::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_uniform && !p.is_slice)
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

        // 2a'. Vertex-index builtins: `vertex_id()` / `instance_id()` load
        // u32 Input variables decorated `BuiltIn VertexIndex` /
        // `BuiltIn InstanceIndex` — the Vulkan-flavoured pair (42/43), NOT
        // the OpenGL VertexId(5)/InstanceId(6). Each is declared only when
        // the body calls it (an unused builtin input just bloats the
        // interface) and carries ONLY the BuiltIn decoration: no Location,
        // and no Flat — Flat belongs to vertex→fragment interpolants, never
        // vertex-stage Inputs. The scan is deterministic over the same body,
        // so the real and passthrough calls declare identically (the
        // id-consistency contract in the doc comment above).
        self.vertex_index_var = None;
        self.instance_index_var = None;
        if super::body_calls(&shader.body_source, "vertex_id") {
            let u32_ty = self.ensure_type_u32();
            let ptr_input_u32 = self.ensure_type_pointer(STORAGE_CLASS_INPUT, u32_ty);
            let var_id = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_input_u32, var_id, STORAGE_CLASS_INPUT],
            );
            self.emit_name(var_id, "gl_VertexIndex");
            self.decorate(var_id, DECORATION_BUILTIN, &[BUILTIN_VERTEX_INDEX]);
            interface_ids.push(var_id);
            self.vertex_index_var = Some(var_id);
        }
        if super::body_calls(&shader.body_source, "instance_id") {
            let u32_ty = self.ensure_type_u32();
            let ptr_input_u32 = self.ensure_type_pointer(STORAGE_CLASS_INPUT, u32_ty);
            let var_id = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_input_u32, var_id, STORAGE_CLASS_INPUT],
            );
            self.emit_name(var_id, "gl_InstanceIndex");
            self.decorate(var_id, DECORATION_BUILTIN, &[BUILTIN_INSTANCE_INDEX]);
            interface_ids.push(var_id);
            self.instance_index_var = Some(var_id);
        }

        // 2b. Declare uniform + slice params as storage-buffer blocks, both
        // drawing from one shared decl-index binding space (see
        // super::shared_binding_indices); the combined-cap error surfaces here.
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

        // 2c. Declare one Output variable per varying FIELD of the shared
        // interface struct (`ShaderDef::varyings`), named by field name, at
        // the Location given by field-declaration order. The body's tail
        // struct literal stores each field explicitly — no auto-forwarding.
        let mut varying_outputs: HashMap<String, (u32, u32, quanta_ir::ShaderType)> =
            HashMap::new();
        if let Some(v) = &shader.varyings {
            for (loc, field) in v.fields.iter().enumerate() {
                let ty_id = self.shader_type_id(field.ty);
                let ptr_ty = self.ensure_type_pointer(STORAGE_CLASS_OUTPUT, ty_id);
                let out_var = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_global_var,
                    OP_VARIABLE,
                    &[ptr_ty, out_var, STORAGE_CLASS_OUTPUT],
                );
                self.emit_name(out_var, &format!("out_{}", field.name));
                self.decorate(out_var, DECORATION_LOCATION, &[loc as u32]);
                // Integer varyings are Flat on BOTH ends of the interpolant
                // (this Output and the fragment's matching Input): integers
                // cannot be interpolated, and the fragment side is invalid
                // without it. A u32 vertex-attribute INPUT stays undecorated —
                // Flat is never valid on vertex-stage Inputs. Float varyings
                // stay smooth.
                if field.ty == quanta_ir::ShaderType::U32 {
                    self.decorate(out_var, DECORATION_FLAT, &[]);
                }
                interface_ids.push(out_var);
                varying_outputs.insert(field.name.clone(), (out_var, ty_id, field.ty));
            }
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
        self.current_block = entry_label;

        // Build param_info
        let mut param_info: Vec<(String, u32, u32, quanta_ir::ShaderType)> = attr_params
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
        // Varying Outputs (if any) stay unwritten — a passthrough is already a
        // declared misrender; the interface itself is identical to the real
        // module by construction.
        if passthrough {
            let result_id =
                self.passthrough_first_input(&attr_params, &input_vars, f32_ty, vec4_ty);
            Self::emit_op(&mut self.sec_function, OP_STORE, &[position_var, result_id]);
            Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
            Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);
            return Ok(super::ShaderEmit::Real);
        }

        // Real attempt: translate the body. A failure interns ids into other
        // sections, so we abandon this emitter and let the caller rebuild the
        // passthrough on a fresh one rather than patching the id state here.
        //
        // Under the shared-struct model the body ends in the Varyings struct
        // literal: the position field stores to gl_Position and every varying
        // field to its named Output. Without varyings the tail expression IS
        // the position (promoted to vec4).
        if let Some(v) = &shader.varyings {
            match self.eval_vertex_varyings_body(
                &shader.body_source,
                &param_info,
                v,
                position_var,
                &varying_outputs,
            ) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!(
                        "[quanta] warning: vertex shader `{}` body failed SPIR-V translation ({e}); \
                         emitting a passthrough SPIR-V shader — it will MISRENDER \
                         on Vulkan (Metal/metallib is unaffected)",
                        shader.name
                    );
                    return Ok(super::ShaderEmit::NeedsPassthrough);
                }
            }
            Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
            Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);
            return Ok(super::ShaderEmit::Real);
        }

        let result_id = match self.eval_shader_body(&shader.body_source, &param_info) {
            Ok((id, ty)) => match self.promote_to_vec4(id, ty, f32_ty, vec4_ty) {
                Some(id) => id,
                None => {
                    eprintln!(
                        "[quanta] warning: vertex shader `{}` body result could not be promoted to Vec4; \
                     emitting a passthrough SPIR-V shader — it will MISRENDER \
                     on Vulkan (Metal/metallib is unaffected)",
                        shader.name
                    );
                    return Ok(super::ShaderEmit::NeedsPassthrough);
                }
            },
            Err(e) => {
                eprintln!(
                    "[quanta] warning: vertex shader `{}` body failed SPIR-V translation ({e}); \
                     emitting a passthrough SPIR-V shader — it will MISRENDER \
                     on Vulkan (Metal/metallib is unaffected)",
                    shader.name
                );
                return Ok(super::ShaderEmit::NeedsPassthrough);
            }
        };

        Self::emit_op(&mut self.sec_function, OP_STORE, &[position_var, result_id]);
        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(super::ShaderEmit::Real)
    }

    /// One storage-buffer block per shader uniform, each at its shared
    /// decl-index binding (`bindings` is parallel to `uniform_params`) —
    /// matching the runtime: `.uniform(slot, …)` binds a STORAGE_BUFFER
    /// descriptor at binding=slot, visible to BOTH stages. Uniform and slice
    /// params draw from ONE binding space (see `shared_binding_indices`), so
    /// the binding is passed in rather than derived from the position among
    /// uniforms alone. The slot space is shared across stages (identical to
    /// Metal, where the runtime binds each slot's buffer to both stages'
    /// `[[buffer(i)]]` index): vertex uniform i and fragment uniform i read the
    /// SAME bound Field. Shared by the vertex and fragment emitters.
    pub(crate) fn emit_uniform_storage_blocks(
        &mut self,
        uniform_params: &[(usize, &quanta_ir::ShaderParam)],
        bindings: &[u32],
        uniform_vars: &mut Vec<(String, u32, u32, quanta_ir::ShaderType)>,
    ) {
        for ((_, p), &binding) in uniform_params.iter().zip(bindings.iter()) {
            let member_ty = self.shader_type_id(p.ty);
            let struct_ty = self.alloc_id();
            Self::emit_op(
                &mut self.sec_type_const,
                OP_TYPE_STRUCT,
                &[struct_ty, member_ty],
            );
            self.decorate(struct_ty, DECORATION_BLOCK, &[]);
            self.member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
            if matches!(
                p.ty,
                quanta_ir::ShaderType::Mat4 | quanta_ir::ShaderType::Mat3
            ) {
                self.member_decorate(struct_ty, 0, 5 /* ColMajor */, &[]);
                self.member_decorate(struct_ty, 0, 7 /* MatrixStride */, &[16]);
            }
            let ptr_ssbo = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, struct_ty);
            let var_id = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_ssbo, var_id, STORAGE_CLASS_STORAGE_BUFFER],
            );
            self.emit_name(var_id, &p.name);
            self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
            self.decorate(var_id, DECORATION_BINDING, &[binding]);
            // A shader uniform is read-only — it is never written by the
            // shader body. Mark the storage-buffer variable NonWritable
            // (same as the slice/FieldRead block below); otherwise a
            // uniform read from a mutable-typed SSBO in either stage trips
            // VUID-RuntimeSpirv-NonWritable-06340/06341 under validation.
            self.decorate(var_id, DECORATION_NON_WRITABLE, &[]);
            uniform_vars.push((p.name.clone(), var_id, member_ty, p.ty));
        }
    }

    /// One read-only runtime-array storage buffer per `&[T]` slice param, at
    /// its shared decl-index binding (`bindings` is parallel to
    /// `slice_params`). Mirrors the compute-kernel `FieldRead` block
    /// (`OpTypeStruct { OpTypeRuntimeArray elem }`, `Block`, `ArrayStride`
    /// 4/8/16, `NonWritable`, DescriptorSet 0), which is the same descriptor
    /// the runtime's `.uniform(slot, &field)` binds. Records each slice in
    /// `self.slice_params` so the body's `name[index]` postfix can access it.
    pub(crate) fn emit_slice_storage_blocks(
        &mut self,
        slice_params: &[(usize, &quanta_ir::ShaderParam)],
        bindings: &[u32],
    ) {
        for ((_, p), &binding) in slice_params.iter().zip(bindings.iter()) {
            let elem_ty = self.shader_type_id(p.ty);
            let stride = match p.ty {
                quanta_ir::ShaderType::F32 => 4,
                quanta_ir::ShaderType::Vec2 => 8,
                quanta_ir::ShaderType::Vec4 => 16,
                // Slice element types are validated at parse time to f32/Vec2/
                // Vec4; treat anything else as tightly-packed vec4 defensively.
                _ => 16,
            };

            let rt_arr = self.ensure_type_runtime_array(elem_ty);
            if self.decorated_stride.insert(rt_arr) {
                self.decorate(rt_arr, DECORATION_ARRAY_STRIDE, &[stride]);
            }

            let struct_ty = self.ensure_type_struct(&[rt_arr]);
            if self.decorated_block.insert(struct_ty) {
                self.decorate(struct_ty, DECORATION_BLOCK, &[]);
                self.member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
            }

            let ptr_struct = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, struct_ty);
            let var_id = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_struct, var_id, STORAGE_CLASS_STORAGE_BUFFER],
            );
            self.emit_name(var_id, &p.name);
            self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
            self.decorate(var_id, DECORATION_BINDING, &[binding]);
            self.decorate(var_id, DECORATION_NON_WRITABLE, &[]);

            self.slice_params
                .insert(p.name.clone(), (var_id, elem_ty, p.ty));
        }
    }
}
