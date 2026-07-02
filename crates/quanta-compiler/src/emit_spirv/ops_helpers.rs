//! Helper methods for KernelOp emission — load/store, binop, unary, cmp, cast.
//!
//! These are called from the main dispatch in ops.rs to keep each method small.

use quanta_ir::*;

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    pub(crate) fn emit_op_load(
        &mut self,
        dst: Reg,
        field: u32,
        index: Reg,
        ty: ScalarType,
    ) -> Result<(), String> {
        let (var_id, elem_ty, _) = *self
            .field_vars
            .get(&field)
            .ok_or_else(|| format!("field {} not declared", field))?;

        let result_ty = self.scalar_type_id(ty);

        let alignment = Self::scalar_byte_size(ty);

        if index.0 == u32::MAX {
            // Push constant: access this slot's member of the shared block
            let member_idx = self.push_constant_member.get(&field).copied().unwrap_or(0);
            let member = self.emit_constant_u32(member_idx);
            let sc = if self.is_push_constant_field(field) {
                STORAGE_CLASS_PUSH_CONSTANT
            } else {
                STORAGE_CLASS_STORAGE_BUFFER
            };
            let ptr_elem = self.ensure_type_pointer(sc, elem_ty);
            let chain = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_ACCESS_CHAIN,
                &[ptr_elem, chain, var_id, member],
            );
            let loaded = self.alloc_id();
            // Memory operand 0x2 = Aligned, followed by alignment value
            Self::emit_op(
                &mut self.sec_function,
                OP_LOAD,
                &[result_ty, loaded, chain, 0x2, alignment],
            );
            self.set_reg(dst, loaded, result_ty);
        } else {
            // Array access: struct member 0, then index into runtime array
            let idx = self.reg_value_id(index)?;
            let zero = self.emit_constant_u32(0);
            let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
            let chain = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_ACCESS_CHAIN,
                &[ptr_elem, chain, var_id, zero, idx],
            );
            let loaded = self.alloc_id();
            // Memory operand 0x2 = Aligned, followed by alignment value
            Self::emit_op(
                &mut self.sec_function,
                OP_LOAD,
                &[result_ty, loaded, chain, 0x2, alignment],
            );
            self.set_reg(dst, loaded, result_ty);
        }
        Ok(())
    }

    pub(crate) fn emit_op_store(
        &mut self,
        field: u32,
        index: Reg,
        src: Reg,
        ty: ScalarType,
    ) -> Result<(), String> {
        let (var_id, elem_ty, _) = *self
            .field_vars
            .get(&field)
            .ok_or_else(|| format!("field {} not declared", field))?;

        let idx = self.reg_value_id(index)?;
        let mut val = self.reg_value_id(src)?;
        // The stored value must match the buffer element type. A bool value
        // (e.g. a compare result flowing into `out[i] = (a < b) as u32`) must be
        // materialized as an int first — OpStore is strictly typed and there is
        // no bool buffer element type. Other type combinations are already
        // reconciled upstream by the producing op.
        let bool_ty = self.ensure_type_bool();
        if elem_ty != bool_ty && self.bool_vals.contains(&val) {
            let as_uint = self.bool_to_int(val);
            let uint_ty = self.ensure_type_u32();
            val = self.coerce_to(as_uint, uint_ty, elem_ty);
        }
        let zero = self.emit_constant_u32(0);
        let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
        let chain = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ACCESS_CHAIN,
            &[ptr_elem, chain, var_id, zero, idx],
        );
        let alignment = Self::scalar_byte_size(ty);
        // Memory operand 0x2 = Aligned, followed by alignment value
        Self::emit_op(
            &mut self.sec_function,
            OP_STORE,
            &[chain, val, 0x2, alignment],
        );
        Ok(())
    }

    pub(crate) fn emit_op_binop(
        &mut self,
        dst: Reg,
        a: Reg,
        b: Reg,
        op: BinOp,
        ty: ScalarType,
    ) -> Result<(), String> {
        let a_val = self.reg_value_id(a)?;
        let b_val = self.reg_value_id(b)?;
        let a_ty = self.reg_type_id(a)?;
        let b_ty = self.reg_type_id(b)?;
        let result_ty = self.scalar_type_id(ty);
        let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
        let is_signed = matches!(
            ty,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        );

        // Rotate ops: SPIR-V has no native rotate. Emit the manual
        // decomposition `(a << k) | (a >> (W - k))` with k masked
        // to [0, W).
        if matches!(op, BinOp::Rotl | BinOp::Rotr) {
            let width: u32 = match ty {
                ScalarType::U8 | ScalarType::I8 => 8,
                ScalarType::U16 | ScalarType::I16 | ScalarType::F16 => 16,
                ScalarType::U32
                | ScalarType::I32
                | ScalarType::F32
                | ScalarType::BF16
                | ScalarType::FP8E5M2
                | ScalarType::FP8E4M3
                | ScalarType::I4 => 32,
                ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => 64,
                ScalarType::Bool => 1,
            };
            let mask = width - 1;
            // For slice-1 surface (i32/u32 rotations) the shift
            // operand width is u32. Future i64 rotations need an
            // emit_constant_u64 + matching width type.
            let mask_val = self.emit_constant_u32(mask);
            let width_val = self.emit_constant_u32(width);
            let k_masked = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_BITWISE_AND,
                &[result_ty, k_masked, b_val, mask_val],
            );
            let w_minus_k = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_ISUB,
                &[result_ty, w_minus_k, width_val, k_masked],
            );
            let w_minus_k_masked = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_BITWISE_AND,
                &[result_ty, w_minus_k_masked, w_minus_k, mask_val],
            );
            let (shl_amt, shr_amt) = if matches!(op, BinOp::Rotl) {
                (k_masked, w_minus_k_masked)
            } else {
                (w_minus_k_masked, k_masked)
            };
            let lo = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_SHIFT_LEFT_LOGICAL,
                &[result_ty, lo, a_val, shl_amt],
            );
            let hi = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_SHIFT_RIGHT_LOGICAL,
                &[result_ty, hi, a_val, shr_amt],
            );
            let result = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_BITWISE_OR,
                &[result_ty, result, lo, hi],
            );
            self.set_reg(dst, result, result_ty);
            return Ok(());
        }

        let opcode = match (op, is_float, is_signed) {
            (BinOp::Add, true, _) => OP_FADD,
            (BinOp::Add, false, _) => OP_IADD,
            (BinOp::Sub, true, _) => OP_FSUB,
            (BinOp::Sub, false, _) => OP_ISUB,
            (BinOp::Mul, true, _) => OP_FMUL,
            (BinOp::Mul, false, _) => OP_IMUL,
            (BinOp::Div, true, _) => OP_FDIV,
            (BinOp::Div, false, true) => OP_SDIV,
            (BinOp::Div, false, false) => OP_UDIV,
            (BinOp::Rem, true, _) => OP_FREM,
            (BinOp::Rem, false, true) => OP_SREM,
            (BinOp::Rem, false, false) => OP_UMOD,
            (BinOp::BitAnd, _, _) => OP_BITWISE_AND,
            (BinOp::BitOr, _, _) => OP_BITWISE_OR,
            (BinOp::BitXor, _, _) => OP_BITWISE_XOR,
            (BinOp::Shl, _, _) => OP_SHIFT_LEFT_LOGICAL,
            (BinOp::Shr, _, true) => OP_SHIFT_RIGHT_ARITHMETIC,
            (BinOp::Shr, _, false) => OP_SHIFT_RIGHT_LOGICAL,
            (BinOp::SatAdd, _, _) | (BinOp::SatSub, _, _) => 0,
            (BinOp::Rotl, _, _) | (BinOp::Rotr, _, _) => unreachable!(),
        };

        if matches!(op, BinOp::SatAdd | BinOp::SatSub) {
            if is_float {
                let base_op = if matches!(op, BinOp::SatAdd) {
                    OP_FADD
                } else {
                    OP_FSUB
                };
                let raw = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    base_op,
                    &[result_ty, raw, a_val, b_val],
                );
                self.set_reg(dst, raw, result_ty);
            } else if matches!(op, BinOp::SatAdd) {
                let sum = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_IADD,
                    &[result_ty, sum, a_val, b_val],
                );
                let bool_ty = self.ensure_type_bool();
                let overflow = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ULESS_THAN,
                    &[bool_ty, overflow, sum, a_val],
                );
                let max_val = self.emit_constant_u32(u32::MAX);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_SELECT,
                    &[result_ty, result, overflow, max_val, sum],
                );
                self.set_reg(dst, result, result_ty);
            } else {
                let bool_ty = self.ensure_type_bool();
                let underflow = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ULESS_THAN,
                    &[bool_ty, underflow, a_val, b_val],
                );
                let diff = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ISUB,
                    &[result_ty, diff, a_val, b_val],
                );
                let zero = self.emit_constant_u32(0);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_SELECT,
                    &[result_ty, result, underflow, zero, diff],
                );
                self.set_reg(dst, result, result_ty);
            }
        } else {
            // Signed int ops (SDiv/SRem/arithmetic-shift-right) need `%int`
            // operands + result; everything else uses the canonical `%uint`.
            // Coerce operands to the op's type and the result back to `%uint`.
            let needs_signed = matches!(
                (op, is_signed),
                (BinOp::Div, true) | (BinOp::Rem, true) | (BinOp::Shr, true)
            );
            let op_ty = if is_float {
                result_ty
            } else if needs_signed {
                self.ensure_type_i32_for(ty)
            } else {
                result_ty
            };
            let a_val = self.coerce_to(a_val, a_ty, op_ty);
            let b_val = self.coerce_to(b_val, b_ty, op_ty);
            let raw = self.alloc_id();
            Self::emit_op(&mut self.sec_function, opcode, &[op_ty, raw, a_val, b_val]);
            let result = self.coerce_to(raw, op_ty, result_ty);
            self.set_reg(dst, result, result_ty);
        }
        Ok(())
    }

    pub(crate) fn emit_op_unary(
        &mut self,
        dst: Reg,
        a: Reg,
        op: UnaryOp,
        ty: ScalarType,
    ) -> Result<(), String> {
        let a_val = self.reg_value_id(a)?;
        let result_ty = self.scalar_type_id(ty);
        let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);

        let result = self.alloc_id();
        match op {
            UnaryOp::Neg if is_float => {
                Self::emit_op(
                    &mut self.sec_function,
                    OP_F_NEGATE,
                    &[result_ty, result, a_val],
                );
            }
            UnaryOp::Neg => {
                Self::emit_op(
                    &mut self.sec_function,
                    OP_S_NEGATE,
                    &[result_ty, result, a_val],
                );
            }
            UnaryOp::BitNot => {
                Self::emit_op(&mut self.sec_function, OP_NOT, &[result_ty, result, a_val]);
            }
            UnaryOp::LogicalNot => {
                let bool_ty = self.ensure_type_bool();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_LOGICAL_NOT,
                    &[bool_ty, result, a_val],
                );
                self.set_reg(dst, result, bool_ty);
                return Ok(());
            }
        }
        self.set_reg(dst, result, result_ty);
        Ok(())
    }

    pub(crate) fn emit_op_cmp(
        &mut self,
        dst: Reg,
        a: Reg,
        b: Reg,
        op: CmpOp,
        ty: ScalarType,
    ) -> Result<(), String> {
        let a_val = self.reg_value_id(a)?;
        let b_val = self.reg_value_id(b)?;
        let a_ty = self.reg_type_id(a)?;
        let b_ty = self.reg_type_id(b)?;
        let bool_ty = self.ensure_type_bool();
        let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
        let is_signed = matches!(
            ty,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        );

        // Operands must share the type the opcode expects: `%int` for a signed
        // compare, `%uint`/float otherwise. First materialize any bool operand as
        // an int (`bool ? 1 : 0`) — the wasm route emits integer compares like
        // `i32.eqz` directly on a compare result — then coerce signedness.
        let operand_ty = if is_float {
            self.scalar_type_id(ty)
        } else if is_signed {
            self.ensure_type_i32_for(ty)
        } else if matches!(ty, ScalarType::Bool) {
            // OpIEqual & friends take *int* operands — a Bool-typed compare
            // lane (wasm `i32.eq` over compare results) compares the 0/1
            // materializations, never `%bool` values directly.
            self.ensure_type_u32()
        } else {
            self.scalar_type_id(ty)
        };
        let (a_val, a_ty) = if !is_float && self.bool_vals.contains(&a_val) {
            (self.bool_to_int(a_val), self.ensure_type_u32())
        } else {
            (a_val, a_ty)
        };
        let (b_val, b_ty) = if !is_float && self.bool_vals.contains(&b_val) {
            (self.bool_to_int(b_val), self.ensure_type_u32())
        } else {
            (b_val, b_ty)
        };
        let a_val = self.coerce_to(a_val, a_ty, operand_ty);
        let b_val = self.coerce_to(b_val, b_ty, operand_ty);

        let opcode = match (op, is_float, is_signed) {
            (CmpOp::Eq, true, _) => OP_FORD_EQUAL,
            (CmpOp::Eq, false, _) => OP_IEQUAL,
            (CmpOp::Ne, true, _) => OP_FORD_NOT_EQUAL,
            (CmpOp::Ne, false, _) => OP_INOT_EQUAL,
            (CmpOp::Lt, true, _) => OP_FORD_LESS_THAN,
            (CmpOp::Lt, false, true) => OP_SLESS_THAN,
            (CmpOp::Lt, false, false) => OP_ULESS_THAN,
            (CmpOp::Le, true, _) => OP_FORD_LESS_THAN_EQUAL,
            (CmpOp::Le, false, true) => OP_SLESS_THAN_EQUAL,
            (CmpOp::Le, false, false) => OP_ULESS_THAN_EQ,
            (CmpOp::Gt, true, _) => OP_FORD_GREATER_THAN,
            (CmpOp::Gt, false, true) => OP_SGREATER_THAN,
            (CmpOp::Gt, false, false) => OP_UGREATER_THAN,
            (CmpOp::Ge, true, _) => OP_FORD_GREATER_THAN_EQUAL,
            (CmpOp::Ge, false, true) => OP_SGREATER_THAN_EQUAL,
            (CmpOp::Ge, false, false) => OP_UGREATER_THAN_EQUAL,
        };

        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            opcode,
            &[bool_ty, result, a_val, b_val],
        );
        self.bool_vals.insert(result);
        self.set_reg(dst, result, bool_ty);
        Ok(())
    }

    pub(crate) fn emit_op_cast(
        &mut self,
        dst: Reg,
        src: Reg,
        from: ScalarType,
        to: ScalarType,
    ) -> Result<(), String> {
        let src_val = self.reg_value_id(src)?;
        let result_ty = self.scalar_type_id(to);

        // OpBitcast to or from %bool is invalid SPIR-V (bools have no bit
        // representation). Bool → int materializes 0/1; int → bool is a
        // truthiness test. Trust `bool_vals` over the declared `from` — the
        // wasm route can reuse a register for both an int and a bool value.
        let src_is_bool = matches!(from, ScalarType::Bool) || self.bool_vals.contains(&src_val);
        if src_is_bool && !matches!(to, ScalarType::Bool) {
            let as_int = self.bool_to_int(src_val);
            let uint_ty = self.ensure_type_u32();
            let result = self.coerce_to(as_int, uint_ty, result_ty);
            self.set_reg(dst, result, result_ty);
            return Ok(());
        }
        if matches!(to, ScalarType::Bool) && !src_is_bool {
            let src_ty = self.reg_type_id(src)?;
            let bool_ty = self.ensure_type_bool();
            let result = self.coerce_to(src_val, src_ty, bool_ty);
            self.bool_vals.insert(result);
            self.set_reg(dst, result, bool_ty);
            return Ok(());
        }

        let from_float = matches!(from, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
        let to_float = matches!(to, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
        let from_signed = matches!(
            from,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        );
        let to_signed = matches!(
            to,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        );

        let result = self.alloc_id();
        let opcode = match (from_float, to_float, from_signed, to_signed) {
            (false, true, true, _) => OP_CONVERT_S_TO_F,
            (false, true, false, _) => OP_CONVERT_U_TO_F,
            (true, false, _, true) => OP_CONVERT_F_TO_S,
            (true, false, _, false) => OP_CONVERT_F_TO_U,
            _ => OP_BITCAST,
        };
        Self::emit_op(
            &mut self.sec_function,
            opcode,
            &[result_ty, result, src_val],
        );
        self.set_reg(dst, result, result_ty);
        Ok(())
    }
}
