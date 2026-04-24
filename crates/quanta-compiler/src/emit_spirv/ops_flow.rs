//! Control flow ops — Branch, Loop, shared memory load/store, math calls.

use quanta_ir::*;

use super::constants::*;
use super::emitter::SpvEmitter;

#[allow(clippy::too_many_arguments)]
impl SpvEmitter {
    pub(crate) fn emit_op_branch(
        &mut self,
        cond: Reg,
        then_ops: &[KernelOp],
        else_ops: &[KernelOp],
        gid_var: u32,
        proton_id_var: u32,
        nucleus_id_var: u32,
        num_wg_var: u32,
    ) -> Result<(), String> {
        let cond_val = self.reg_value_id(cond)?;
        let then_label = self.alloc_id();
        let else_label = self.alloc_id();
        let merge_label = self.alloc_id();

        // OpSelectionMerge
        Self::emit_op(
            &mut self.sec_function,
            OP_SELECTION_MERGE,
            &[merge_label, SELECTION_CONTROL_NONE],
        );

        if else_ops.is_empty() {
            Self::emit_op(
                &mut self.sec_function,
                OP_BRANCH_CONDITIONAL,
                &[cond_val, then_label, merge_label],
            );
        } else {
            Self::emit_op(
                &mut self.sec_function,
                OP_BRANCH_CONDITIONAL,
                &[cond_val, then_label, else_label],
            );
        }

        // Then block
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[then_label]);
        self.emit_ops(then_ops, gid_var, proton_id_var, nucleus_id_var, num_wg_var)?;
        Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

        // Else block
        if !else_ops.is_empty() {
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[else_label]);
            self.emit_ops(else_ops, gid_var, proton_id_var, nucleus_id_var, num_wg_var)?;
            Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);
        }

        // Merge block
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);
        Ok(())
    }

    pub(crate) fn emit_op_loop(
        &mut self,
        count: Reg,
        iter_reg: Reg,
        body: &[KernelOp],
        gid_var: u32,
        proton_id_var: u32,
        nucleus_id_var: u32,
        num_wg_var: u32,
    ) -> Result<(), String> {
        let count_val = self.reg_value_id(count)?;
        let uint_ty = self.ensure_type_u32();

        // Detect loop-carried registers
        let written_in_body = Self::collect_dsts(body);
        let mut carried: Vec<(u32, u32, u32)> = Vec::new();
        for &reg_num in &written_in_body {
            if let Some(&pre_id) = self.reg_ids.get(&reg_num)
                && let Some(&ty_id) = self.reg_types.get(&reg_num)
            {
                carried.push((reg_num, pre_id, ty_id));
            }
        }

        let pre_header_label = self.alloc_id();
        let header_label = self.alloc_id();
        let cond_label = self.alloc_id();
        let body_label = self.alloc_id();
        let continue_label = self.alloc_id();
        let merge_label = self.alloc_id();

        let zero = self.emit_constant_u32(0);
        let one = self.emit_constant_u32(1);

        // Pre-header: branch to header
        Self::emit_op(&mut self.sec_function, OP_BRANCH, &[pre_header_label]);
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[pre_header_label]);
        Self::emit_op(&mut self.sec_function, OP_BRANCH, &[header_label]);

        // Header block
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[header_label]);

        // OpPhi for the loop counter
        let phi_id = self.alloc_id();
        let inc_id = self.alloc_id(); // forward-reference
        Self::emit_op(
            &mut self.sec_function,
            OP_PHI,
            &[
                uint_ty,
                phi_id,
                zero,
                pre_header_label,
                inc_id,
                continue_label,
            ],
        );
        self.set_reg(iter_reg, phi_id, uint_ty);

        // OpPhi for each loop-carried variable
        let mut carried_phis: Vec<(u32, u32, u32, u32)> = Vec::new();
        for &(reg_num, pre_id, ty_id) in &carried {
            let header_phi = self.alloc_id();
            let continue_copy = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_PHI,
                &[
                    ty_id,
                    header_phi,
                    pre_id,
                    pre_header_label,
                    continue_copy,
                    continue_label,
                ],
            );
            self.set_reg(Reg(reg_num), header_phi, ty_id);
            carried_phis.push((reg_num, header_phi, continue_copy, ty_id));
        }

        // OpLoopMerge
        Self::emit_op(
            &mut self.sec_function,
            OP_LOOP_MERGE,
            &[merge_label, continue_label, LOOP_CONTROL_NONE],
        );

        // OpBranch to condition block
        Self::emit_op(&mut self.sec_function, OP_BRANCH, &[cond_label]);

        // Condition block
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[cond_label]);
        let bool_ty = self.ensure_type_bool();
        let cond = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ULESS_THAN,
            &[bool_ty, cond, phi_id, count_val],
        );
        Self::emit_op(
            &mut self.sec_function,
            OP_BRANCH_CONDITIONAL,
            &[cond, body_label, merge_label],
        );

        // Body block
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[body_label]);
        self.loop_merge_stack.push(merge_label);
        self.emit_ops(body, gid_var, proton_id_var, nucleus_id_var, num_wg_var)?;
        self.loop_merge_stack.pop();
        Self::emit_op(&mut self.sec_function, OP_BRANCH, &[continue_label]);

        // Continue block: copy carried values, increment counter
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[continue_label]);
        for &(reg_num, _header_phi, continue_copy, ty_id) in &carried_phis {
            let current = self.reg_ids[&reg_num];
            Self::emit_op(
                &mut self.sec_function,
                OP_COPY_OBJECT,
                &[ty_id, continue_copy, current],
            );
        }
        Self::emit_op(
            &mut self.sec_function,
            OP_IADD,
            &[uint_ty, inc_id, phi_id, one],
        );
        Self::emit_op(&mut self.sec_function, OP_BRANCH, &[header_label]);

        // Merge block
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);

        for &(reg_num, header_phi, _, ty_id) in &carried_phis {
            self.set_reg(Reg(reg_num), header_phi, ty_id);
        }
        Ok(())
    }

    pub(crate) fn emit_op_shared_load(
        &mut self,
        dst: Reg,
        id: u32,
        index: Reg,
        ty: ScalarType,
    ) -> Result<(), String> {
        let (var_id, elem_ty) = *self
            .shared_vars
            .get(&id)
            .ok_or_else(|| format!("shared memory {} not declared", id))?;
        let idx = self.reg_value_id(index)?;
        let result_ty = self.scalar_type_id(ty);

        let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, elem_ty);
        let chain = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ACCESS_CHAIN,
            &[ptr_elem, chain, var_id, idx],
        );
        let loaded = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LOAD, &[result_ty, loaded, chain]);
        self.set_reg(dst, loaded, result_ty);
        Ok(())
    }

    pub(crate) fn emit_op_shared_store(
        &mut self,
        id: u32,
        index: Reg,
        src: Reg,
    ) -> Result<(), String> {
        let (var_id, elem_ty) = *self
            .shared_vars
            .get(&id)
            .ok_or_else(|| format!("shared memory {} not declared", id))?;
        let idx = self.reg_value_id(index)?;
        let val = self.reg_value_id(src)?;

        let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, elem_ty);
        let chain = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ACCESS_CHAIN,
            &[ptr_elem, chain, var_id, idx],
        );
        Self::emit_op(&mut self.sec_function, OP_STORE, &[chain, val]);
        Ok(())
    }

    pub(crate) fn emit_op_math_call(
        &mut self,
        dst: Reg,
        func: MathFn,
        args: &[Reg],
        ty: ScalarType,
    ) -> Result<(), String> {
        let ext_id = self.ensure_glsl_ext();
        let result_ty = self.scalar_type_id(ty);
        let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
        let is_signed = matches!(
            ty,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        );

        let glsl_op = match func {
            MathFn::Sin => GLSL_SIN,
            MathFn::Cos => GLSL_COS,
            MathFn::Tan => GLSL_TAN,
            MathFn::Asin => GLSL_ASIN,
            MathFn::Acos => GLSL_ACOS,
            MathFn::Atan => GLSL_ATAN,
            MathFn::Atan2 => GLSL_ATAN2,
            MathFn::Sqrt => GLSL_SQRT,
            MathFn::Rsqrt => GLSL_INVERSE_SQRT,
            MathFn::Exp => GLSL_EXP,
            MathFn::Exp2 => GLSL_EXP2,
            MathFn::Log => GLSL_LOG,
            MathFn::Log2 => GLSL_LOG2,
            MathFn::Pow => GLSL_POW,
            MathFn::Abs if is_float => GLSL_FABS,
            MathFn::Abs if is_signed => GLSL_SABS,
            MathFn::Abs => GLSL_FABS,
            MathFn::Min if is_float => GLSL_FMIN,
            MathFn::Min if is_signed => GLSL_SMIN,
            MathFn::Min => GLSL_UMIN,
            MathFn::Max if is_float => GLSL_FMAX,
            MathFn::Max if is_signed => GLSL_SMAX,
            MathFn::Max => GLSL_UMAX,
            MathFn::Clamp if is_float => GLSL_FCLAMP,
            MathFn::Clamp if is_signed => GLSL_SCLAMP,
            MathFn::Clamp => GLSL_UCLAMP,
            MathFn::Floor => GLSL_FLOOR,
            MathFn::Ceil => GLSL_CEIL,
            MathFn::Round => GLSL_ROUND,
            MathFn::Fma => GLSL_FMA,
        };

        let mut operand_ids = Vec::with_capacity(args.len());
        for arg in args {
            operand_ids.push(self.reg_value_id(*arg)?);
        }

        let result = self.alloc_id();
        let mut ops = vec![result_ty, result, ext_id, glsl_op];
        ops.extend_from_slice(&operand_ids);
        Self::emit_op(&mut self.sec_function, OP_EXT_INST, &ops);
        self.set_reg(dst, result, result_ty);
        Ok(())
    }

    pub(crate) fn emit_op_device_call(
        &mut self,
        dst: Reg,
        func_name: &str,
        args: &[Reg],
        ty: ScalarType,
    ) -> Result<(), String> {
        if let Some((fn_id, ret_ty, _param_tys)) = self.device_fn_ids.get(func_name).cloned() {
            let result = self.alloc_id();
            let mut operands = vec![ret_ty, result, fn_id];
            for arg in args {
                let arg_id = self.reg_value_id(*arg)?;
                operands.push(arg_id);
            }
            Self::emit_op(&mut self.sec_function, OP_FUNCTION_CALL, &operands);
            self.set_reg(dst, result, ret_ty);
        } else {
            let result_ty = self.scalar_type_id(ty);
            let zero = match ty {
                ScalarType::F32 | ScalarType::F16 => self.emit_constant_f32(0.0),
                ScalarType::F64 => self.emit_constant_f64(0.0),
                ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 => {
                    self.emit_constant_i32(0)
                }
                _ => self.emit_constant_u32(0),
            };
            self.set_reg(dst, zero, result_ty);
        }
        Ok(())
    }
}
