//! Kernel parameter setup — storage buffers, push constants, textures.

use quanta_ir::KernelParam;

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    /// Emit kernel parameter declarations (storage buffers, push constants, textures).
    pub(crate) fn emit_kernel_params(&mut self, params: &[KernelParam]) -> Result<(), String> {
        // Vulkan allows at most ONE push-constant block per entry point, so
        // scalar `Constant` params are gathered into a single Block struct
        // (one member per param, offset = slot*16 — the runtime pushes one
        // blob with each slot at a 16-byte-aligned offset). The previous
        // one-variable-per-constant emission violated
        // VUID-StandaloneSpirv-OpEntryPoint-06674 and, worse, same-typed
        // constants shared one cached struct type so only the first slot's
        // Offset decoration ever landed.
        let mut constants: Vec<(String, u32, quanta_ir::ScalarType)> = Vec::new();
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

                    let rt_arr = self.ensure_type_runtime_array(elem_ty);
                    if self.decorated_stride.insert(rt_arr) {
                        self.decorate(rt_arr, DECORATION_ARRAY_STRIDE, &[stride]);
                    }

                    let struct_ty = self.ensure_type_struct(&[rt_arr]);
                    if self.decorated_block.insert(struct_ty) {
                        self.decorate(struct_ty, DECORATION_BLOCK, &[]);
                        self.member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
                    }

                    let ptr_struct =
                        self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, struct_ty);

                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_struct, var_id, STORAGE_CLASS_STORAGE_BUFFER],
                    );
                    self.emit_name(var_id, name);

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
                KernelParam::Texture2DWrite { name, slot, .. } => {
                    self.emit_texture_2d_write(name, *slot);
                }
                KernelParam::Texture3DRead { name, slot, .. } => {
                    self.emit_texture_3d_read(name, *slot);
                }
            }
        }
        self.emit_push_constant_block(&constants);
        Ok(())
    }

    /// Emit the single push-constant Block for all scalar `Constant` params.
    /// Member `i` (in slot order) sits at byte offset `slot*16`, matching the
    /// runtime's inline push buffer layout (`Wave::set_value`).
    fn emit_push_constant_block(&mut self, constants: &[(String, u32, quanta_ir::ScalarType)]) {
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

        for (i, &(_, slot, sty)) in constants.iter().enumerate() {
            let elem_ty = self.push_constant_type_id(sty);
            self.field_vars.insert(slot, (var_id, elem_ty, false));
            self.push_constant_slots.insert(slot);
            self.push_constant_member.insert(slot, i as u32);
            self.push_constant_size += 16;
        }
    }

    pub(crate) fn emit_texture_2d_read(&mut self, name: &str, slot: u32) {
        let f32_ty = self.ensure_type_f32();
        let image_ty = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_IMAGE,
            &[
                image_ty, f32_ty, 1, /*Dim2D*/
                0, 0, 0, 1, /*sampled*/
                0,
            ],
        );
        let sampled_image_ty = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_SAMPLED_IMAGE,
            &[sampled_image_ty, image_ty],
        );
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
    }

    pub(crate) fn emit_texture_2d_write(&mut self, name: &str, slot: u32) {
        let f32_ty = self.ensure_type_f32();
        let image_ty = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_IMAGE,
            &[
                image_ty, f32_ty, 1, 0, 0, 0, 2, /*storage*/
                3, /*Rgba32f*/
            ],
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
    }

    pub(crate) fn emit_texture_3d_read(&mut self, name: &str, slot: u32) {
        let f32_ty = self.ensure_type_f32();
        let image_ty = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_IMAGE,
            &[image_ty, f32_ty, 2 /*Dim3D*/, 0, 0, 0, 1, 0],
        );
        let sampled_image_ty = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_SAMPLED_IMAGE,
            &[sampled_image_ty, image_ty],
        );
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
    }
}
