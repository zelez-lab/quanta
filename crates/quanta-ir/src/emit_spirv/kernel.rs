//! Kernel entry point — emit a full compute kernel SPIR-V module.
//!
//! Sets up built-in variables (GlobalInvocationId, LocalInvocationId, etc.),
//! storage buffers, push constants, shared memory, and the main function body.

use crate::*;

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    /// Check if any ops in the kernel body use subgroup operations.
    pub(crate) fn uses_subgroup_ops(ops: &[KernelOp]) -> bool {
        for op in ops {
            match op {
                KernelOp::SubgroupReduceAdd { .. }
                | KernelOp::SubgroupReduceMin { .. }
                | KernelOp::SubgroupReduceMax { .. }
                | KernelOp::SubgroupExclusiveAdd { .. }
                | KernelOp::SubgroupInclusiveAdd { .. } => return true,
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    if Self::uses_subgroup_ops(then_ops) || Self::uses_subgroup_ops(else_ops) {
                        return true;
                    }
                }
                KernelOp::Loop { body, .. } => {
                    if Self::uses_subgroup_ops(body) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    pub(crate) fn emit_kernel(&mut self, kernel: &KernelDef) -> Result<(), String> {
        // 1. Capability
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_SHADER],
        );

        // Add subgroup capabilities if needed
        if Self::uses_subgroup_ops(&kernel.body) {
            Self::emit_op(
                &mut self.sec_capability,
                OP_CAPABILITY,
                &[CAPABILITY_GROUP_NON_UNIFORM],
            );
            Self::emit_op(
                &mut self.sec_capability,
                OP_CAPABILITY,
                &[CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC],
            );
        }

        // 2. Memory model
        Self::emit_op(
            &mut self.sec_memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        // 3. Set up built-in: GlobalInvocationId
        let v3uint = self.ensure_type_v3uint();
        let ptr_input_v3uint = self.ensure_type_pointer(STORAGE_CLASS_INPUT, v3uint);
        let gid_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_input_v3uint, gid_var, STORAGE_CLASS_INPUT],
        );
        self.emit_name(gid_var, "gl_GlobalInvocationId");
        self.decorate(gid_var, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

        // LocalInvocationId
        let proton_id_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_input_v3uint, proton_id_var, STORAGE_CLASS_INPUT],
        );
        self.emit_name(proton_id_var, "gl_LocalInvocationId");
        self.decorate(
            proton_id_var,
            DECORATION_BUILTIN,
            &[BUILTIN_LOCAL_INVOCATION_ID],
        );

        // WorkgroupId
        let nucleus_id_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_input_v3uint, nucleus_id_var, STORAGE_CLASS_INPUT],
        );
        self.emit_name(nucleus_id_var, "gl_WorkGroupID");
        self.decorate(nucleus_id_var, DECORATION_BUILTIN, &[BUILTIN_WORKGROUP_ID]);

        // NumWorkgroups
        let num_wg_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_input_v3uint, num_wg_var, STORAGE_CLASS_INPUT],
        );
        self.emit_name(num_wg_var, "gl_NumWorkGroups");
        self.decorate(num_wg_var, DECORATION_BUILTIN, &[BUILTIN_NUM_WORKGROUPS]);

        // Collect Input/Output interface variables for the entry point.
        // SPIR-V 1.3 requires only Input/Output variables in the interface list.
        // StorageBuffer, Uniform, and Workgroup variables must NOT be listed.
        let interface_ids = vec![gid_var, proton_id_var, nucleus_id_var, num_wg_var];

        // 4. Set up storage buffers for each field parameter
        self.emit_kernel_params(&kernel.params)?;

        // 5. Scan body for SharedDecl and emit workgroup variables
        self.emit_shared_decls(&kernel.body)?;

        // 5b. Emit device functions as SPIR-V OpFunction definitions.
        // Done before the main function so function IDs are available.
        self.emit_device_functions(kernel, gid_var, proton_id_var, nucleus_id_var, num_wg_var)?;

        // 6. Entry point (Input/Output variables only in SPIR-V 1.3)
        let void_ty = self.ensure_type_void();
        let func_ty = self.ensure_type_function(void_ty, &[]);
        let main_id = self.alloc_id();
        self.emit_name(main_id, "main");

        {
            let name_words = Self::string_words("main");
            let mut ops = vec![EXECUTION_MODEL_GLCOMPUTE, main_id];
            ops.extend_from_slice(&name_words);
            ops.extend_from_slice(&interface_ids);
            Self::emit_op(&mut self.sec_entry_point, OP_ENTRY_POINT, &ops);
        }

        // 7. Execution mode: LocalSize from kernel workgroup_size
        Self::emit_op(
            &mut self.sec_execution_mode,
            OP_EXECUTION_MODE,
            &[
                main_id,
                EXECUTION_MODE_LOCAL_SIZE,
                kernel.workgroup_size[0],
                kernel.workgroup_size[1],
                kernel.workgroup_size[2],
            ],
        );

        // 8. Function body
        // OpFunction
        Self::emit_op(
            &mut self.sec_function,
            OP_FUNCTION,
            &[void_ty, main_id, FUNCTION_CONTROL_NONE, func_ty],
        );

        let entry_label = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[entry_label]);

        // Emit the body ops
        self.emit_ops(
            &kernel.body,
            gid_var,
            proton_id_var,
            nucleus_id_var,
            num_wg_var,
        )?;

        // OpReturn + OpFunctionEnd
        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(())
    }

    /// Set up storage buffers and push constants for each kernel parameter.
    fn emit_kernel_params(&mut self, params: &[KernelParam]) -> Result<(), String> {
        for param in params {
            match param {
                KernelParam::FieldRead {
                    name,
                    slot,
                    scalar_type,
                }
                | KernelParam::FieldWrite {
                    name,
                    slot,
                    scalar_type,
                } => {
                    let is_writable = matches!(param, KernelParam::FieldWrite { .. });
                    let elem_ty = self.scalar_type_id(*scalar_type);
                    let stride = Self::scalar_byte_size(*scalar_type);

                    // RuntimeArray of element type
                    let rt_arr = self.ensure_type_runtime_array(elem_ty);
                    if self.decorated_stride.insert(rt_arr) {
                        self.decorate(rt_arr, DECORATION_ARRAY_STRIDE, &[stride]);
                    }

                    // Struct wrapping the runtime array
                    let struct_ty = self.ensure_type_struct(&[rt_arr]);
                    if self.decorated_block.insert(struct_ty) {
                        self.decorate(struct_ty, DECORATION_BLOCK, &[]);
                        self.member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
                    }

                    // Pointer to struct in StorageBuffer
                    let ptr_struct =
                        self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, struct_ty);

                    // Variable
                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_struct, var_id, STORAGE_CLASS_STORAGE_BUFFER],
                    );
                    self.emit_name(var_id, name);

                    // Decorations
                    self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
                    self.decorate(var_id, DECORATION_BINDING, &[*slot]);
                    if !is_writable {
                        self.decorate(var_id, DECORATION_NON_WRITABLE, &[]);
                        self.decorate(var_id, DECORATION_RESTRICT, &[]);
                    }

                    self.field_vars
                        .insert(*slot, (var_id, elem_ty, is_writable));
                }
                KernelParam::Constant {
                    name,
                    slot,
                    scalar_type,
                } => {
                    let elem_ty = self.scalar_type_id(*scalar_type);
                    // Push constants: wrap in a struct with Block decoration,
                    // use PushConstant storage class (matches vkCmdPushConstants).
                    let struct_ty = self.ensure_type_struct(&[elem_ty]);
                    if self.decorated_block.insert(struct_ty) {
                        self.decorate(struct_ty, DECORATION_BLOCK, &[]);
                        self.member_decorate(struct_ty, 0, DECORATION_OFFSET, &[*slot * 16]);
                    }

                    let ptr_struct =
                        self.ensure_type_pointer(STORAGE_CLASS_PUSH_CONSTANT, struct_ty);

                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_struct, var_id, STORAGE_CLASS_PUSH_CONSTANT],
                    );
                    self.emit_name(var_id, name);
                    // PushConstant doesn't use DescriptorSet/Binding — it's accessed
                    // via the push constant range in the pipeline layout.

                    // Store as field_vars — Load with index=MAX will access member 0
                    self.field_vars.insert(*slot, (var_id, elem_ty, false));
                    self.push_constant_slots.insert(*slot);
                    self.push_constant_size += 16;
                }
                _ => {
                    // Texture params — not yet supported in SPIR-V emitter
                }
            }
        }
        Ok(())
    }

    /// Collect all register numbers written (as dst) in a sequence of ops.
    /// Used to detect loop-carried variables.
    pub(crate) fn collect_dsts(ops: &[KernelOp]) -> Vec<u32> {
        let mut dsts = Vec::new();
        for op in ops {
            match op {
                KernelOp::QuarkId { dst }
                | KernelOp::ProtonId { dst }
                | KernelOp::NucleusId { dst }
                | KernelOp::QuarkCount { dst }
                | KernelOp::ProtonSize { dst }
                | KernelOp::Const { dst, .. }
                | KernelOp::Load { dst, .. }
                | KernelOp::BinOp { dst, .. }
                | KernelOp::UnaryOp { dst, .. }
                | KernelOp::Cmp { dst, .. }
                | KernelOp::Cast { dst, .. }
                | KernelOp::Copy { dst, .. }
                | KernelOp::SharedLoad { dst, .. }
                | KernelOp::MathCall { dst, .. }
                | KernelOp::AtomicOp { dst, .. }
                | KernelOp::AtomicCas { dst, .. }
                | KernelOp::WaveShuffle { dst, .. }
                | KernelOp::WaveBallot { dst, .. }
                | KernelOp::WaveAny { dst, .. }
                | KernelOp::WaveAll { dst, .. }
                | KernelOp::VecConstruct { dst, .. }
                | KernelOp::VecExtract { dst, .. }
                | KernelOp::MatMul { dst, .. }
                | KernelOp::DeviceCall { dst, .. }
                | KernelOp::TextureSample2D { dst, .. }
                | KernelOp::TextureSample3D { dst, .. }
                | KernelOp::Bitcast { dst, .. }
                | KernelOp::CountTrailingZeros { dst, .. }
                | KernelOp::CountLeadingZeros { dst, .. }
                | KernelOp::PopCount { dst, .. }
                | KernelOp::Dot { dst, .. }
                | KernelOp::SubgroupReduceAdd { dst, .. }
                | KernelOp::SubgroupReduceMin { dst, .. }
                | KernelOp::SubgroupReduceMax { dst, .. }
                | KernelOp::SubgroupExclusiveAdd { dst, .. }
                | KernelOp::SubgroupInclusiveAdd { dst, .. }
                | KernelOp::TextureLoad2D { dst, .. }
                | KernelOp::SubgroupSize { dst, .. } => {
                    dsts.push(dst.0);
                }
                KernelOp::TextureSize { dst_w, dst_h, .. } => {
                    dsts.push(dst_w.0);
                    dsts.push(dst_h.0);
                }
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    dsts.extend(Self::collect_dsts(then_ops));
                    dsts.extend(Self::collect_dsts(else_ops));
                }
                KernelOp::Loop { body, iter_reg, .. } => {
                    dsts.push(iter_reg.0);
                    dsts.extend(Self::collect_dsts(body));
                }
                _ => {}
            }
        }
        dsts.sort_unstable();
        dsts.dedup();
        dsts
    }

    /// Scan for SharedDecl ops and create workgroup variables.
    pub(crate) fn emit_shared_decls(&mut self, ops: &[KernelOp]) -> Result<(), String> {
        for op in ops {
            match op {
                KernelOp::SharedDecl { id, ty, count } => {
                    let elem_ty = self.scalar_type_id(*ty);
                    let count_const = self.emit_constant_u32(*count);
                    let arr_ty = self.ensure_type_array(elem_ty, count_const);
                    let stride = Self::scalar_byte_size(*ty);
                    if self.decorated_stride.insert(arr_ty) {
                        self.decorate(arr_ty, DECORATION_ARRAY_STRIDE, &[stride]);
                    }
                    let ptr_arr = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, arr_ty);
                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_arr, var_id, STORAGE_CLASS_WORKGROUP],
                    );
                    self.emit_name(var_id, &format!("shared_{}", id));
                    self.shared_vars.insert(*id, (var_id, elem_ty));
                }
                KernelOp::SharedDeclDyn { id, ty } => {
                    let elem_ty = self.scalar_type_id(*ty);
                    let default_count = self.emit_constant_u32(256);
                    let arr_ty = self.ensure_type_array(elem_ty, default_count);
                    let stride = Self::scalar_byte_size(*ty);
                    if self.decorated_stride.insert(arr_ty) {
                        self.decorate(arr_ty, DECORATION_ARRAY_STRIDE, &[stride]);
                    }
                    let ptr_arr = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, arr_ty);
                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_arr, var_id, STORAGE_CLASS_WORKGROUP],
                    );
                    self.emit_name(var_id, &format!("shared_dyn_{}", id));
                    self.shared_vars.insert(*id, (var_id, elem_ty));
                }
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    self.emit_shared_decls(then_ops)?;
                    self.emit_shared_decls(else_ops)?;
                }
                KernelOp::Loop { body, .. } => {
                    self.emit_shared_decls(body)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Emit a sequence of KernelOps into the function body.
    pub(crate) fn emit_ops(
        &mut self,
        ops: &[KernelOp],
        gid_var: u32,
        proton_id_var: u32,
        nucleus_id_var: u32,
        num_wg_var: u32,
    ) -> Result<(), String> {
        for op in ops {
            self.emit_single_op(op, gid_var, proton_id_var, nucleus_id_var, num_wg_var)?;
        }
        Ok(())
    }

    /// Load a built-in vec3<u32> and extract .x component.
    pub(crate) fn load_builtin_x(&mut self, var_id: u32) -> u32 {
        let v3uint = self.ensure_type_v3uint();
        let uint_ty = self.ensure_type_u32();

        let loaded = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LOAD, &[v3uint, loaded, var_id]);

        let x_val = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_COMPOSITE_EXTRACT,
            &[uint_ty, x_val, loaded, 0],
        );

        x_val
    }

    /// Emit device functions as SPIR-V OpFunction definitions.
    /// Must be called before emit_kernel so that function IDs are available
    /// for OpFunctionCall during body emission. The function bodies are
    /// emitted into sec_device_fns which is placed before sec_function
    /// in the final module.
    fn emit_device_functions(
        &mut self,
        kernel: &KernelDef,
        gid_var: u32,
        proton_id_var: u32,
        nucleus_id_var: u32,
        num_wg_var: u32,
    ) -> Result<(), String> {
        for device_fn in &kernel.device_functions {
            let ret_ty = self.scalar_type_id(device_fn.return_type);

            // Build parameter type IDs
            let mut param_type_ids = Vec::new();
            for (_name, ty) in &device_fn.params {
                param_type_ids.push(self.scalar_type_id(*ty));
            }

            // Create function type: OpTypeFunction ret_ty param_types...
            let func_type = self.ensure_type_function(ret_ty, &param_type_ids);

            // Allocate function ID
            let fn_id = self.alloc_id();
            self.emit_name(fn_id, &device_fn.name);

            // Store mapping for OpFunctionCall
            self.device_fn_ids.insert(
                device_fn.name.clone(),
                (fn_id, ret_ty, param_type_ids.clone()),
            );

            // OpFunction
            Self::emit_op(
                &mut self.sec_device_fns,
                OP_FUNCTION,
                &[ret_ty, fn_id, FUNCTION_CONTROL_NONE, func_type],
            );

            // OpFunctionParameter for each param — save the register mapping
            let old_reg_ids = self.reg_ids.clone();
            let old_reg_types = self.reg_types.clone();
            self.reg_ids.clear();
            self.reg_types.clear();

            for (i, (pname, ty)) in device_fn.params.iter().enumerate() {
                let param_id = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_device_fns,
                    OP_FUNCTION_PARAMETER,
                    &[param_type_ids[i], param_id],
                );
                self.emit_name(param_id, pname);
                // Map the register that the parser assigned to this param
                // The parser allocates registers 0..N for N params
                let type_id = self.scalar_type_id(*ty);
                self.reg_ids.insert(i as u32, param_id);
                self.reg_types.insert(i as u32, type_id);
            }

            // OpLabel for the function body
            let body_label = self.alloc_id();
            Self::emit_op(&mut self.sec_device_fns, OP_LABEL, &[body_label]);

            // Emit the function body into a temporary buffer, then move to sec_device_fns
            let saved_fn = std::mem::take(&mut self.sec_function);
            self.emit_ops(
                &device_fn.body,
                gid_var,
                proton_id_var,
                nucleus_id_var,
                num_wg_var,
            )?;

            // Find the last value produced — that's the return value.
            // The device function body should leave the result in the last
            // register written. We look for it by scanning the reg_ids for
            // the highest register number that was set during body emission.
            let return_val = self.find_return_value(&device_fn.body, device_fn.return_type);

            let body_words = std::mem::replace(&mut self.sec_function, saved_fn);
            self.sec_device_fns.extend_from_slice(&body_words);

            // OpReturnValue with the return value
            if let Some(ret_id) = return_val {
                Self::emit_op(
                    &mut self.sec_device_fns,
                    252, // OpReturnValue
                    &[ret_id],
                );
            } else {
                Self::emit_op(&mut self.sec_device_fns, OP_RETURN, &[]);
            }

            Self::emit_op(&mut self.sec_device_fns, OP_FUNCTION_END, &[]);

            // Restore main function's register context
            self.reg_ids = old_reg_ids;
            self.reg_types = old_reg_types;
        }
        Ok(())
    }

    /// Find the SPIR-V ID of the return value for a device function body.
    /// The last expression in the body determines the return value.
    fn find_return_value(&self, ops: &[KernelOp], _ret_ty: ScalarType) -> Option<u32> {
        // Walk backwards to find the last op that writes to a dst register
        for op in ops.iter().rev() {
            let dst_reg = match op {
                KernelOp::BinOp { dst, .. }
                | KernelOp::UnaryOp { dst, .. }
                | KernelOp::Cmp { dst, .. }
                | KernelOp::Cast { dst, .. }
                | KernelOp::Const { dst, .. }
                | KernelOp::Load { dst, .. }
                | KernelOp::SharedLoad { dst, .. }
                | KernelOp::MathCall { dst, .. }
                | KernelOp::Copy { dst, .. }
                | KernelOp::DeviceCall { dst, .. }
                | KernelOp::Bitcast { dst, .. }
                | KernelOp::CountTrailingZeros { dst, .. }
                | KernelOp::CountLeadingZeros { dst, .. }
                | KernelOp::PopCount { dst, .. }
                | KernelOp::Dot { dst, .. }
                | KernelOp::SubgroupReduceAdd { dst, .. }
                | KernelOp::SubgroupReduceMin { dst, .. }
                | KernelOp::SubgroupReduceMax { dst, .. }
                | KernelOp::SubgroupExclusiveAdd { dst, .. }
                | KernelOp::SubgroupInclusiveAdd { dst, .. }
                | KernelOp::TextureLoad2D { dst, .. }
                | KernelOp::SubgroupSize { dst, .. } => Some(dst.0),
                _ => None,
            };
            if let Some(reg_num) = dst_reg
                && let Some(&id) = self.reg_ids.get(&reg_num)
            {
                return Some(id);
            }
        }
        None
    }
}
