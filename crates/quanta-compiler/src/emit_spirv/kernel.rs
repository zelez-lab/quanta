//! Kernel entry point — emit a full compute kernel SPIR-V module.
//!
//! Sets up built-in variables (GlobalInvocationId, LocalInvocationId, etc.),
//! storage buffers, push constants, textures, shared memory, and the main
//! function body.

use quanta_ir::{KernelDef, KernelOp};

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
                } if Self::uses_subgroup_ops(then_ops) || Self::uses_subgroup_ops(else_ops) => {
                    return true;
                }
                KernelOp::Loop { body, .. } if Self::uses_subgroup_ops(body) => {
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    pub(crate) fn emit_kernel(&mut self, kernel: &KernelDef) -> Result<(), String> {
        // A storage image (write-declared slot) cannot be sampled — reject
        // before emitting anything so the error is the same on every backend.
        quanta_ir::types::reject_sample_on_write(kernel)?;
        // A sampled `&Texture2D<u32>` is not wired (storage-position u32 is the
        // packed-RGBA8 image); reject it rather than emit a float sampled image.
        quanta_ir::types::reject_sampled_u32_texture(kernel)?;

        // Record wg_x for the folded-dispatch linearization constant
        // in QuarkId (see load_linear builtin helper).
        self.wg_x = kernel.workgroup_size[0].max(1);

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

        // Demote mutable registers (written more than once, or written in a
        // Branch arm / Loop body and read past the merge) to Function-storage
        // variables. SPIR-V requires the OpVariables in the first block.
        let demoted = quanta_ir::reg_mutability::collect_mutable_regs(&[], &kernel.body);
        self.declare_demoted_regs(&demoted);

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
                | KernelOp::SubgroupSize { dst, .. }
                | KernelOp::CooperativeMMA { dst, .. } => {
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

    /// Load a built-in vec3<u32> and compute the folded-dispatch
    /// linear index `v.x + v.y * row_span`. `row_span` is the fixed
    /// per-row element count of a folded 1D dispatch
    /// (`FOLD_ROW_GROUPS * wg_x` for GlobalInvocationId,
    /// `FOLD_ROW_GROUPS` for WorkgroupId — see
    /// `quanta_ir::dispatch_fold`). For ordinary 1D dispatches
    /// `v.y == 0`, so the result is exactly the old `.x` read.
    pub(crate) fn load_builtin_linear(&mut self, var_id: u32, row_span: u32) -> u32 {
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
        let y_val = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_COMPOSITE_EXTRACT,
            &[uint_ty, y_val, loaded, 1],
        );
        let span_const = self.emit_constant_u32(row_span);
        let y_scaled = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_IMUL,
            &[uint_ty, y_scaled, y_val, span_const],
        );
        let linear = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_IADD,
            &[uint_ty, linear, x_val, y_scaled],
        );

        linear
    }
}
