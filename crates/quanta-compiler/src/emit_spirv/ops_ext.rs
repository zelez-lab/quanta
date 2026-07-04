//! Extended ops — atomics, wave/subgroup, texture, device call helpers.

use quanta_ir::*;

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_op_atomic(
        &mut self,
        dst: Reg,
        field: u32,
        index: Reg,
        val: Reg,
        op: AtomicOp,
        ty: ScalarType,
        order: MemoryOrder,
    ) -> Result<(), String> {
        let (var_id, elem_ty, _) = *self
            .field_vars
            .get(&field)
            .ok_or_else(|| format!("field {} not declared", field))?;
        let idx = self.reg_value_id(index)?;
        let val_id = self.reg_value_id(val)?;
        let result_ty = self.scalar_type_id(ty);
        let zero = self.emit_constant_u32(0);
        let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
        let chain = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ACCESS_CHAIN,
            &[ptr_elem, chain, var_id, zero, idx],
        );

        let scope = self.emit_constant_u32(1); // Device
        let order_bits: u32 = match order {
            MemoryOrder::Relaxed => 0,
            MemoryOrder::Acquire => MEMORY_SEMANTICS_ACQUIRE,
            MemoryOrder::Release => MEMORY_SEMANTICS_RELEASE,
            MemoryOrder::AcqRel => MEMORY_SEMANTICS_ACQ_REL,
            MemoryOrder::SeqCst => MEMORY_SEMANTICS_SEQ_CST,
        };
        let semantics = self.emit_constant_u32(order_bits | MEMORY_SEMANTICS_WORKGROUP);

        let is_signed = matches!(
            ty,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        );
        let atomic_opcode = match op {
            AtomicOp::Add => OP_ATOMIC_IADD,
            AtomicOp::Sub => OP_ATOMIC_ISUB,
            AtomicOp::Min if is_signed => OP_ATOMIC_SMIN,
            AtomicOp::Min => OP_ATOMIC_UMIN,
            AtomicOp::Max if is_signed => OP_ATOMIC_SMAX,
            AtomicOp::Max => OP_ATOMIC_UMAX,
            AtomicOp::And => OP_ATOMIC_AND,
            AtomicOp::Or => OP_ATOMIC_OR,
            AtomicOp::Xor => OP_ATOMIC_XOR,
            AtomicOp::Exchange => OP_ATOMIC_EXCHANGE,
            AtomicOp::CompareExchange => OP_ATOMIC_COMPARE_EXCHANGE,
        };

        let result_id = self.alloc_id();
        if matches!(op, AtomicOp::CompareExchange) {
            Self::emit_op(
                &mut self.sec_function,
                atomic_opcode,
                &[
                    result_ty, result_id, chain, scope, semantics, semantics, val_id, val_id,
                ],
            );
        } else {
            Self::emit_op(
                &mut self.sec_function,
                atomic_opcode,
                &[result_ty, result_id, chain, scope, semantics, val_id],
            );
        }
        self.set_reg(dst, result_id, result_ty);
        Ok(())
    }

    /// Shared-memory atomic: the buffer atomic above with a
    /// Workgroup-storage pointer (plain array — no struct wrapper,
    /// so no leading zero in the access chain) and Workgroup scope
    /// (2) instead of Device (1).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_op_shared_atomic(
        &mut self,
        dst: Reg,
        slot: u32,
        index: Reg,
        val: Reg,
        op: AtomicOp,
        ty: ScalarType,
        order: MemoryOrder,
    ) -> Result<(), String> {
        let (var_id, elem_ty) = *self
            .shared_vars
            .get(&slot)
            .ok_or_else(|| format!("shared memory {} not declared", slot))?;
        let idx = self.reg_value_id(index)?;
        let val_id = self.reg_value_id(val)?;
        let result_ty = self.scalar_type_id(ty);
        let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, elem_ty);
        let chain = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ACCESS_CHAIN,
            &[ptr_elem, chain, var_id, idx],
        );

        let scope = self.emit_constant_u32(2); // Workgroup
        let order_bits: u32 = match order {
            MemoryOrder::Relaxed => 0,
            MemoryOrder::Acquire => MEMORY_SEMANTICS_ACQUIRE,
            MemoryOrder::Release => MEMORY_SEMANTICS_RELEASE,
            MemoryOrder::AcqRel => MEMORY_SEMANTICS_ACQ_REL,
            MemoryOrder::SeqCst => MEMORY_SEMANTICS_SEQ_CST,
        };
        let semantics = self.emit_constant_u32(order_bits | MEMORY_SEMANTICS_WORKGROUP);

        let is_signed = matches!(
            ty,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        );
        let atomic_opcode = match op {
            AtomicOp::Add => OP_ATOMIC_IADD,
            AtomicOp::Sub => OP_ATOMIC_ISUB,
            AtomicOp::Min if is_signed => OP_ATOMIC_SMIN,
            AtomicOp::Min => OP_ATOMIC_UMIN,
            AtomicOp::Max if is_signed => OP_ATOMIC_SMAX,
            AtomicOp::Max => OP_ATOMIC_UMAX,
            AtomicOp::And => OP_ATOMIC_AND,
            AtomicOp::Or => OP_ATOMIC_OR,
            AtomicOp::Xor => OP_ATOMIC_XOR,
            AtomicOp::Exchange => OP_ATOMIC_EXCHANGE,
            AtomicOp::CompareExchange => OP_ATOMIC_COMPARE_EXCHANGE,
        };

        let result_id = self.alloc_id();
        if matches!(op, AtomicOp::CompareExchange) {
            Self::emit_op(
                &mut self.sec_function,
                atomic_opcode,
                &[
                    result_ty, result_id, chain, scope, semantics, semantics, val_id, val_id,
                ],
            );
        } else {
            Self::emit_op(
                &mut self.sec_function,
                atomic_opcode,
                &[result_ty, result_id, chain, scope, semantics, val_id],
            );
        }
        self.set_reg(dst, result_id, result_ty);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_op_atomic_cas(
        &mut self,
        dst: Reg,
        field: u32,
        index: Reg,
        expected: Reg,
        desired: Reg,
        ty: ScalarType,
        order: MemoryOrder,
    ) -> Result<(), String> {
        let (var_id, elem_ty, _) = *self
            .field_vars
            .get(&field)
            .ok_or_else(|| format!("field {} not declared", field))?;
        let idx = self.reg_value_id(index)?;
        let exp_val = self.reg_value_id(expected)?;
        let des_val = self.reg_value_id(desired)?;
        let result_ty = self.scalar_type_id(ty);
        let zero = self.emit_constant_u32(0);
        let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
        let chain = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ACCESS_CHAIN,
            &[ptr_elem, chain, var_id, zero, idx],
        );

        let scope = self.emit_constant_u32(1); // Device
        let order_bits: u32 = match order {
            MemoryOrder::Relaxed => 0,
            MemoryOrder::Acquire => MEMORY_SEMANTICS_ACQUIRE,
            MemoryOrder::Release => MEMORY_SEMANTICS_RELEASE,
            MemoryOrder::AcqRel => MEMORY_SEMANTICS_ACQ_REL,
            MemoryOrder::SeqCst => MEMORY_SEMANTICS_SEQ_CST,
        };
        let semantics = self.emit_constant_u32(order_bits | MEMORY_SEMANTICS_WORKGROUP);

        // OpAtomicCompareExchange: result_type result pointer scope
        //   equal_sem unequal_sem value comparator
        let result_id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ATOMIC_COMPARE_EXCHANGE,
            &[
                result_ty, result_id, chain, scope, semantics, semantics, des_val, exp_val,
            ],
        );
        self.set_reg(dst, result_id, result_ty);
        Ok(())
    }

    pub(crate) fn emit_op_wave_shuffle(
        &mut self,
        dst: Reg,
        src: Reg,
        lane_delta: Reg,
        ty: ScalarType,
    ) -> Result<(), String> {
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE],
        );
        let src_val = self.reg_value_id(src)?;
        let delta_val = self.reg_value_id(lane_delta)?;
        let result_ty = self.scalar_type_id(ty);
        let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_GROUP_NON_UNIFORM_SHUFFLE,
            &[result_ty, result, scope, src_val, delta_val],
        );
        self.set_reg(dst, result, result_ty);
        Ok(())
    }

    pub(crate) fn emit_op_wave_ballot(&mut self, dst: Reg, predicate: Reg) -> Result<(), String> {
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_GROUP_NON_UNIFORM_BALLOT],
        );
        let pred_val = self.reg_value_id(predicate)?;
        let uint_ty = self.ensure_type_u32();
        let vec4_uint = self.ensure_type_vector(uint_ty, 4);
        let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
        let ballot = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_GROUP_NON_UNIFORM_BALLOT,
            &[vec4_uint, ballot, scope, pred_val],
        );
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_COMPOSITE_EXTRACT,
            &[uint_ty, result, ballot, 0],
        );
        self.set_reg(dst, result, uint_ty);
        Ok(())
    }

    pub(crate) fn emit_op_wave_any(&mut self, dst: Reg, predicate: Reg) -> Result<(), String> {
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_GROUP_NON_UNIFORM],
        );
        // OpGroupNonUniformAny additionally requires GroupNonUniformVote.
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_GROUP_NON_UNIFORM_VOTE],
        );
        let pred_val = self.reg_value_id(predicate)?;
        let bool_ty = self.ensure_type_bool();
        let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
        let result_bool = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_GROUP_NON_UNIFORM_ANY,
            &[bool_ty, result_bool, scope, pred_val],
        );
        let uint_ty = self.ensure_type_u32();
        let one = self.emit_constant_u32(1);
        let zero = self.emit_constant_u32(0);
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_SELECT,
            &[uint_ty, result, result_bool, one, zero],
        );
        self.set_reg(dst, result, uint_ty);
        Ok(())
    }

    pub(crate) fn emit_op_wave_all(&mut self, dst: Reg, predicate: Reg) -> Result<(), String> {
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_GROUP_NON_UNIFORM],
        );
        // OpGroupNonUniformAll additionally requires GroupNonUniformVote.
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_GROUP_NON_UNIFORM_VOTE],
        );
        let pred_val = self.reg_value_id(predicate)?;
        let bool_ty = self.ensure_type_bool();
        let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
        let result_bool = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_GROUP_NON_UNIFORM_ALL,
            &[bool_ty, result_bool, scope, pred_val],
        );
        let uint_ty = self.ensure_type_u32();
        let one = self.emit_constant_u32(1);
        let zero = self.emit_constant_u32(0);
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_SELECT,
            &[uint_ty, result, result_bool, one, zero],
        );
        self.set_reg(dst, result, uint_ty);
        Ok(())
    }

    pub(crate) fn emit_op_texture_sample_2d(
        &mut self,
        dst: Reg,
        texture: u32,
        x: Reg,
        y: Reg,
        ty: ScalarType,
    ) -> Result<(), String> {
        if let Some(&(var_id, type_id)) = self.texture_samplers.get(&texture) {
            let loaded = self.alloc_id();
            Self::emit_op(&mut self.sec_function, OP_LOAD, &[type_id, loaded, var_id]);
            let f32_ty = self.ensure_type_f32();
            let vec2_ty = self.ensure_type_vector(f32_ty, 2);
            let x_val = self.reg_value_id(x)?;
            let y_val = self.reg_value_id(y)?;
            let coord = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_COMPOSITE_CONSTRUCT,
                &[vec2_ty, coord, x_val, y_val],
            );
            let vec4_ty = self.ensure_type_vector(f32_ty, 4);
            let sample_result = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_IMAGE_SAMPLE_IMPLICIT_LOD,
                &[vec4_ty, sample_result, loaded, coord],
            );
            let result_ty = self.scalar_type_id(ty);
            let result = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_COMPOSITE_EXTRACT,
                &[result_ty, result, sample_result, 0],
            );
            self.set_reg(dst, result, result_ty);
        } else {
            let result_ty = self.scalar_type_id(ty);
            let zero = self.emit_constant_f32(0.0);
            self.set_reg(dst, zero, result_ty);
        }
        Ok(())
    }

    pub(crate) fn emit_op_texture_write_2d(
        &mut self,
        texture: u32,
        x: Reg,
        y: Reg,
        value: Reg,
    ) -> Result<(), String> {
        if let Some(&(var_id, type_id)) = self.texture_samplers.get(&texture) {
            let loaded = self.alloc_id();
            Self::emit_op(&mut self.sec_function, OP_LOAD, &[type_id, loaded, var_id]);
            // If the slot was declared OpTypeSampledImage, unwrap to the plain
            // OpTypeImage that OpImageWrite requires (mirrors the read path).
            let image = if let Some(&image_ty) = self.texture_image_types.get(&texture) {
                let image = self.alloc_id();
                Self::emit_op(&mut self.sec_function, OP_IMAGE, &[image_ty, image, loaded]);
                image
            } else {
                loaded
            };
            let int_ty = self.ensure_type_i32();
            let vec2_int = self.ensure_type_vector(int_ty, 2);
            // Coords arrive as %uint (quark_id arithmetic); OpCompositeConstruct
            // requires constituents to match the result vector's component type,
            // so coerce to %int first (same as emit_op_texture_load_2d).
            let x_val = self.reg_value_id(x)?;
            let x_ty = self.reg_type_id(x)?;
            let x_val = self.coerce_to(x_val, x_ty, int_ty);
            let y_val = self.reg_value_id(y)?;
            let y_ty = self.reg_type_id(y)?;
            let y_val = self.coerce_to(y_val, y_ty, int_ty);
            let coord = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_COMPOSITE_CONSTRUCT,
                &[vec2_int, coord, x_val, y_val],
            );
            let f32_ty = self.ensure_type_f32();
            let vec4_ty = self.ensure_type_vector(f32_ty, 4);
            // The texel must be a vec4<f32>; coerce the scalar value to f32 so
            // its constituent type matches the vector's components.
            let val = self.reg_value_id(value)?;
            let val_ty = self.reg_type_id(value)?;
            let val = self.coerce_to(val, val_ty, f32_ty);
            let zero = self.emit_constant_f32(0.0);
            let one = self.emit_constant_f32(1.0);
            let texel = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_COMPOSITE_CONSTRUCT,
                &[vec4_ty, texel, val, zero, zero, one],
            );
            Self::emit_op(
                &mut self.sec_function,
                OP_IMAGE_WRITE,
                &[image, coord, texel],
            );
        }
        Ok(())
    }

    pub(crate) fn emit_op_texture_load_2d(
        &mut self,
        dst: Reg,
        texture: u32,
        x: Reg,
        y: Reg,
        ty: ScalarType,
    ) -> Result<(), String> {
        if let Some(&(var_id, type_id)) = self.texture_samplers.get(&texture) {
            let loaded = self.alloc_id();
            Self::emit_op(&mut self.sec_function, OP_LOAD, &[type_id, loaded, var_id]);
            // Read slots are declared as OpTypeSampledImage, but
            // OpImageFetch takes a plain OpTypeImage — unwrap with OpImage.
            let image = if let Some(&image_ty) = self.texture_image_types.get(&texture) {
                let image = self.alloc_id();
                Self::emit_op(&mut self.sec_function, OP_IMAGE, &[image_ty, image, loaded]);
                image
            } else {
                loaded
            };
            let int_ty = self.ensure_type_i32();
            let vec2_int = self.ensure_type_vector(int_ty, 2);
            // Coordinates typically arrive as %uint (quark_id arithmetic);
            // OpCompositeConstruct requires constituents to match the result
            // vector's component type exactly, so bitcast to %int first.
            let x_val = self.reg_value_id(x)?;
            let x_ty = self.reg_type_id(x)?;
            let x_val = self.coerce_to(x_val, x_ty, int_ty);
            let y_val = self.reg_value_id(y)?;
            let y_ty = self.reg_type_id(y)?;
            let y_val = self.coerce_to(y_val, y_ty, int_ty);
            let coord = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_COMPOSITE_CONSTRUCT,
                &[vec2_int, coord, x_val, y_val],
            );
            let f32_ty = self.ensure_type_f32();
            let vec4_ty = self.ensure_type_vector(f32_ty, 4);
            let fetch_result = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_IMAGE_FETCH,
                &[vec4_ty, fetch_result, image, coord],
            );
            let result_ty = self.scalar_type_id(ty);
            let result = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_COMPOSITE_EXTRACT,
                &[result_ty, result, fetch_result, 0],
            );
            self.set_reg(dst, result, result_ty);
        } else {
            let result_ty = self.scalar_type_id(ty);
            let zero = self.emit_constant_f32(0.0);
            self.set_reg(dst, zero, result_ty);
        }
        Ok(())
    }

    pub(crate) fn emit_op_subgroup_reduce(
        &mut self,
        dst: Reg,
        src: Reg,
        ty: ScalarType,
        exclusive: bool,
        inclusive: bool,
    ) -> Result<(), String> {
        let src_val = self.reg_value_id(src)?;
        let result_ty = self.scalar_type_id(ty);
        let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
        let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
        let opcode = if is_float {
            OP_GROUP_NON_UNIFORM_FADD
        } else {
            OP_GROUP_NON_UNIFORM_IADD
        };
        let group_op = if exclusive {
            GROUP_OPERATION_EXCLUSIVE_SCAN
        } else if inclusive {
            GROUP_OPERATION_INCLUSIVE_SCAN
        } else {
            GROUP_OPERATION_REDUCE
        };
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            opcode,
            &[result_ty, result, scope, group_op, src_val],
        );
        self.set_reg(dst, result, result_ty);
        Ok(())
    }

    pub(crate) fn emit_op_subgroup_minmax(
        &mut self,
        dst: Reg,
        src: Reg,
        ty: ScalarType,
        is_min: bool,
    ) -> Result<(), String> {
        let src_val = self.reg_value_id(src)?;
        let result_ty = self.scalar_type_id(ty);
        let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
        let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
        let is_signed = matches!(
            ty,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        );
        let opcode = match (is_min, is_float, is_signed) {
            (true, true, _) => OP_GROUP_NON_UNIFORM_FMIN,
            (true, false, true) => OP_GROUP_NON_UNIFORM_SMIN,
            (true, false, false) => OP_GROUP_NON_UNIFORM_UMIN,
            (false, true, _) => OP_GROUP_NON_UNIFORM_FMAX,
            (false, false, true) => OP_GROUP_NON_UNIFORM_SMAX,
            (false, false, false) => OP_GROUP_NON_UNIFORM_UMAX,
        };
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            opcode,
            &[result_ty, result, scope, GROUP_OPERATION_REDUCE, src_val],
        );
        self.set_reg(dst, result, result_ty);
        Ok(())
    }
}
