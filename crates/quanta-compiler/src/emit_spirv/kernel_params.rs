//! Kernel parameter setup — storage buffers, push constants, textures.

use quanta_ir::KernelParam;

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    /// Emit kernel parameter declarations (storage buffers, push constants, textures).
    pub(crate) fn emit_kernel_params(&mut self, params: &[KernelParam]) -> Result<(), String> {
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
                    let elem_ty = self.scalar_type_id(*scalar_type);
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

                    self.field_vars.insert(*slot, (var_id, elem_ty, false));
                    self.push_constant_slots.insert(*slot);
                    self.push_constant_size += 16;
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
        Ok(())
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
