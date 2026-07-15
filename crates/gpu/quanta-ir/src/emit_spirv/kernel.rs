//! Kernel entry point — emit a full compute kernel SPIR-V module.
//!
//! Sets up built-in variables (GlobalInvocationId, LocalInvocationId, etc.),
//! storage buffers, push constants, shared memory, and the main function body.

use crate::*;

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    /// Check if any ops in the kernel body use subgroup *arithmetic*
    /// operations (reduce/scan), which require the
    /// GroupNonUniformArithmetic capability.
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

    /// Check if any ops use subgroup shuffle, which requires the
    /// GroupNonUniformShuffle capability (distinct from Arithmetic).
    pub(crate) fn uses_subgroup_shuffle(ops: &[KernelOp]) -> bool {
        for op in ops {
            match op {
                KernelOp::WaveShuffle { .. } => return true,
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } if Self::uses_subgroup_shuffle(then_ops)
                    || Self::uses_subgroup_shuffle(else_ops) =>
                {
                    return true;
                }
                KernelOp::Loop { body, .. } if Self::uses_subgroup_shuffle(body) => {
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    pub(crate) fn emit_kernel(&mut self, kernel: &KernelDef) -> Result<(), String> {
        // A storage image (write-declared slot) cannot be sampled — reject
        // before emitting anything so the error matches the other backends.
        crate::types::reject_sample_on_write(kernel)?;
        // A sampled `&Texture2D<u32>` is not wired (storage-position u32 is the
        // packed-RGBA8 image); reject it rather than emit a float sampled image.
        crate::types::reject_sampled_u32_texture(kernel)?;

        // Record wg_x for the folded-dispatch linearization constant
        // in QuarkId (see load_linear_thread_id).
        self.wg_x = kernel.workgroup_size[0].max(1);

        // 1. Capability
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_SHADER],
        );

        // Add subgroup capabilities if needed. Arithmetic (reduce/scan)
        // and shuffle are separate SPIR-V capabilities; both build on the
        // base GroupNonUniform capability.
        let uses_arith = Self::uses_subgroup_ops(&kernel.body);
        let uses_shuffle = Self::uses_subgroup_shuffle(&kernel.body);
        if uses_arith || uses_shuffle {
            Self::emit_op(
                &mut self.sec_capability,
                OP_CAPABILITY,
                &[CAPABILITY_GROUP_NON_UNIFORM],
            );
        }
        if uses_arith {
            Self::emit_op(
                &mut self.sec_capability,
                OP_CAPABILITY,
                &[CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC],
            );
        }
        if uses_shuffle {
            Self::emit_op(
                &mut self.sec_capability,
                OP_CAPABILITY,
                &[CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE],
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
        let demoted = crate::reg_mutability::collect_mutable_regs(&[], &kernel.body);
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

    /// Set up storage buffers and push constants for each kernel parameter.
    ///
    /// Vulkan allows at most ONE push-constant block per entry point, so
    /// scalar `Constant` params are gathered into a single Block struct (one
    /// member per param, offset = slot*16 — the runtime pushes one blob with
    /// each slot at a 16-byte-aligned offset). The previous
    /// one-variable-per-constant emission violated
    /// VUID-StandaloneSpirv-OpEntryPoint-06674 and, worse, same-typed
    /// constants shared one cached struct type so only the first slot's
    /// Offset decoration ever landed.
    fn emit_kernel_params(&mut self, params: &[KernelParam]) -> Result<(), String> {
        let mut constants: Vec<(String, u32, ScalarType)> = Vec::new();
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
                    let elem_ty = self.storage_scalar_type_id(*scalar_type);
                    let stride = self.storage_byte_size(*scalar_type);

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
                    constants.push((name.clone(), *slot, *scalar_type));
                }
                KernelParam::Texture2DRead { name, slot, .. } => {
                    self.emit_texture_2d_read(name, *slot);
                }
                KernelParam::Texture2DWrite {
                    name,
                    slot,
                    scalar_type,
                } => {
                    self.emit_texture_2d_write(name, *slot, *scalar_type)?;
                }
                KernelParam::Texture3DRead { name, slot, .. } => {
                    self.emit_texture_3d_read(name, *slot);
                }
            }
        }
        self.emit_push_constant_block(&constants);
        Ok(())
    }

    /// Sampled 2D image (`&Texture2D`): OpTypeSampledImage over a float
    /// OpTypeImage. Reads unwrap to the plain image with OpImage before
    /// OpImageFetch (see `emit_op` TextureLoad2D).
    pub(crate) fn emit_texture_2d_read(&mut self, name: &str, slot: u32) {
        let f32_ty = self.ensure_type_f32();
        // Deduped: two `&Texture2D<f32>` params must share one OpTypeImage /
        // OpTypeSampledImage — SPIR-V forbids duplicate non-aggregate types.
        let image_ty = self.ensure_type_image(
            f32_ty, 1, /*Dim2D*/
            0, 0, 0, 1, /*sampled=1*/
            0, /*ImageFormat Unknown*/
        );
        let sampled_image_ty = self.ensure_type_sampled_image(image_ty);
        let ptr_si = self.ensure_type_pointer(STORAGE_CLASS_UNIFORM_CONSTANT, sampled_image_ty);
        let var_id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_si, var_id, STORAGE_CLASS_UNIFORM_CONSTANT],
        );
        self.emit_name(var_id, name);
        self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
        self.decorate(var_id, DECORATION_BINDING, &[slot]);
        self.texture_samplers
            .insert(slot, (var_id, sampled_image_ty));
        self.texture_image_types.insert(slot, image_ty);
    }

    /// Storage 2D image (`&mut Texture2D`): plain OpTypeImage, sampled=2, with
    /// a scalar-driven ImageFormat. The format contract is `Texture2D<f32>` ⇔
    /// R32Float and `Texture2D<u32>` ⇔ Rgba8 (packed-u32 RGBA8-unorm); other
    /// scalars error. Writes emit OpImageWrite; loads emit OpImageRead (a
    /// storage image is not sampled).
    pub(crate) fn emit_texture_2d_write(
        &mut self,
        name: &str,
        slot: u32,
        scalar_type: ScalarType,
    ) -> Result<(), String> {
        let f32_ty = self.ensure_type_f32();
        let image_format = scalar_type.spirv_storage_image_format().ok_or_else(|| {
            format!(
                "storage texture slot {slot} has scalar type {scalar_type:?}; only \
                 Texture2D<f32> (R32Float) storage images are supported"
            )
        })?;
        // Deduped: the src+dst `&mut Texture2D<u32>` ping-pong pair emits two
        // same-shaped storage images; they must share one OpTypeImage or
        // spirv-val rejects the duplicate non-aggregate type declaration.
        let image_ty = self.ensure_type_image(
            f32_ty,
            1, /*Dim2D*/
            0,
            0,
            0,
            2, /*sampled=2: storage image, read_write*/
            image_format,
        );
        let ptr_img = self.ensure_type_pointer(STORAGE_CLASS_UNIFORM_CONSTANT, image_ty);
        let var_id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_img, var_id, STORAGE_CLASS_UNIFORM_CONSTANT],
        );
        self.emit_name(var_id, name);
        self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
        self.decorate(var_id, DECORATION_BINDING, &[slot]);
        self.texture_samplers.insert(slot, (var_id, image_ty));
        self.texture_storage_slots.insert(slot);
        Ok(())
    }

    /// Sampled 3D image (`&Texture3D`): read-only/sampled, unchanged semantics.
    pub(crate) fn emit_texture_3d_read(&mut self, name: &str, slot: u32) {
        let f32_ty = self.ensure_type_f32();
        // Deduped, same rationale as the 2D read path.
        let image_ty = self.ensure_type_image(f32_ty, 2 /*Dim3D*/, 0, 0, 0, 1, 0);
        let sampled_image_ty = self.ensure_type_sampled_image(image_ty);
        let ptr_si = self.ensure_type_pointer(STORAGE_CLASS_UNIFORM_CONSTANT, sampled_image_ty);
        let var_id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_si, var_id, STORAGE_CLASS_UNIFORM_CONSTANT],
        );
        self.emit_name(var_id, name);
        self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
        self.decorate(var_id, DECORATION_BINDING, &[slot]);
        self.texture_samplers
            .insert(slot, (var_id, sampled_image_ty));
        self.texture_image_types.insert(slot, image_ty);
    }

    /// Emit the single push-constant Block for all scalar `Constant` params.
    /// Member `i` (in slot order) sits at byte offset `slot*16`, matching the
    /// runtime's inline push buffer layout (`Wave::set_value`).
    fn emit_push_constant_block(&mut self, constants: &[(String, u32, ScalarType)]) {
        if constants.is_empty() {
            return;
        }
        let mut constants = constants.to_vec();
        constants.sort_by_key(|&(_, slot, _)| slot);

        let member_tys: Vec<u32> = constants
            .iter()
            .map(|&(_, _, sty)| self.push_constant_type_id(sty))
            .collect();
        let struct_ty = self.ensure_type_struct(&member_tys);
        if self.decorated_block.insert(struct_ty) {
            self.decorate(struct_ty, DECORATION_BLOCK, &[]);
            for (i, &(_, slot, _)) in constants.iter().enumerate() {
                self.member_decorate(struct_ty, i as u32, DECORATION_OFFSET, &[slot * 16]);
            }
        }

        let ptr_struct = self.ensure_type_pointer(STORAGE_CLASS_PUSH_CONSTANT, struct_ty);
        let var_id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_struct, var_id, STORAGE_CLASS_PUSH_CONSTANT],
        );
        self.emit_name(var_id, "push_constants");
        // PushConstant doesn't use DescriptorSet/Binding — it's accessed
        // via the push constant range in the pipeline layout.

        for (i, &(_, slot, sty)) in constants.iter().enumerate() {
            let elem_ty = self.push_constant_type_id(sty);
            // Store as field_vars — Load with index=MAX accesses this
            // slot's member of the shared block.
            self.field_vars.insert(slot, (var_id, elem_ty, false));
            self.push_constant_slots.insert(slot);
            self.push_constant_member.insert(slot, i as u32);
            self.push_constant_size += 16;
        }
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
            let old_demoted = std::mem::take(&mut self.demoted_regs);
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

            // Mutable-register pre-pass for the device function body; the
            // params count as a first write at the outermost scope.
            let pre_written: Vec<(u32, crate::ScalarType)> = device_fn
                .params
                .iter()
                .enumerate()
                .map(|(i, (_, ty))| (i as u32, *ty))
                .collect();
            let demoted =
                crate::reg_mutability::collect_mutable_regs(&pre_written, &device_fn.body);

            // Emit the function body (entry label first, so the demoted
            // OpVariables land in the function's first block) into a
            // temporary buffer, then move it to sec_device_fns.
            let saved_fn = std::mem::take(&mut self.sec_function);
            let body_label = self.alloc_id();
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[body_label]);
            self.declare_demoted_regs(&demoted);
            // A demoted param's incoming value seeds its variable; drop the
            // SSA mapping so later reads go through the variable.
            for (i, (_, ty)) in device_fn.params.iter().enumerate() {
                let reg = i as u32;
                if demoted.contains_key(&reg)
                    && let Some(param_id) = self.reg_ids.remove(&reg)
                {
                    let param_ty = self.scalar_type_id(*ty);
                    self.set_reg(crate::Reg(reg), param_id, param_ty);
                }
            }
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
            self.demoted_regs = old_demoted;
        }
        Ok(())
    }

    /// Find the SPIR-V ID of the return value for a device function body.
    /// The last expression in the body determines the return value. For a
    /// demoted (mutable) register this emits an `OpLoad` of its variable, so
    /// it must run while `sec_function` still holds the device fn body.
    fn find_return_value(&mut self, ops: &[KernelOp], _ret_ty: ScalarType) -> Option<u32> {
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
            if let Some(reg_num) = dst_reg {
                if self.demoted_regs.contains_key(&reg_num) {
                    return self.reg_value_id(Reg(reg_num)).ok();
                }
                if let Some(&id) = self.reg_ids.get(&reg_num) {
                    return Some(id);
                }
            }
        }
        None
    }
}
