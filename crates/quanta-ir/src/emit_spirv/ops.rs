//! KernelOp → SPIR-V instruction dispatch.
//!
//! The single `emit_single_op` method handles every KernelOp variant,
//! emitting the corresponding SPIR-V instruction(s) into sec_function.

use crate::*;

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    pub(crate) fn emit_single_op(
        &mut self,
        op: &KernelOp,
        gid_var: u32,
        proton_id_var: u32,
        nucleus_id_var: u32,
        num_wg_var: u32,
    ) -> Result<(), String> {
        match op {
            KernelOp::QuarkId { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.load_builtin_x(gid_var);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::ProtonId { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.load_builtin_x(proton_id_var);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::NucleusId { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.load_builtin_x(nucleus_id_var);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::QuarkCount { dst } => {
                // num_workgroups.x * workgroup_size (64)
                let uint_ty = self.ensure_type_u32();
                let nwg = self.load_builtin_x(num_wg_var);
                let sixty_four = self.emit_constant_u32(64);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_IMUL,
                    &[uint_ty, result, nwg, sixty_four],
                );
                self.set_reg(*dst, result, uint_ty);
            }

            KernelOp::ProtonSize { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.emit_constant_u32(64);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::Const { dst, value } => {
                let (id, ty) = match value {
                    ConstValue::F32(v) => {
                        let ty = self.ensure_type_f32();
                        (self.emit_constant_f32(*v), ty)
                    }
                    ConstValue::F64(v) => {
                        let ty = self.ensure_type_f64();
                        (self.emit_constant_f64(*v), ty)
                    }
                    ConstValue::U32(v) => {
                        let ty = self.ensure_type_u32();
                        (self.emit_constant_u32(*v), ty)
                    }
                    ConstValue::U64(v) => {
                        let ty = self.ensure_type_u64();
                        (self.emit_constant_u64(*v), ty)
                    }
                    ConstValue::I32(v) => {
                        let ty = self.ensure_type_i32();
                        (self.emit_constant_i32(*v), ty)
                    }
                    ConstValue::I64(v) => {
                        let ty = self.ensure_type_i64();
                        (self.emit_constant_i64(*v), ty)
                    }
                    ConstValue::Bool(v) => {
                        let ty = self.ensure_type_bool();
                        (self.emit_constant_bool(*v), ty)
                    }
                    ConstValue::F16(v) => {
                        // Convert F16 to F32
                        let ty = self.ensure_type_f32();
                        let f = f32::from_bits((*v as u32) << 16);
                        (self.emit_constant_f32(f), ty)
                    }
                };
                self.set_reg(*dst, id, ty);
                // Track integer constants so the Loop emitter can pick
                // LOOP_CONTROL_UNROLL for short iteration counts (TODO T1405).
                // Floats and booleans aren't tracked — they don't feed
                // Loop.count.
                match value {
                    ConstValue::U32(v) => {
                        self.reg_const_int.insert(dst.0, *v as i64);
                    }
                    ConstValue::U64(v) => {
                        // U64 truncates to u32 in the emit above; track
                        // the truncated value to match what the SPIR-V
                        // type system sees.
                        self.reg_const_int.insert(dst.0, (*v as u32) as i64);
                    }
                    ConstValue::I32(v) => {
                        self.reg_const_int.insert(dst.0, *v as i64);
                    }
                    ConstValue::I64(v) => {
                        // I64 truncates to i32 above; same logic.
                        self.reg_const_int.insert(dst.0, (*v as i32) as i64);
                    }
                    _ => {}
                }
            }

            KernelOp::Load {
                dst,
                field,
                index,
                ty,
            } => {
                let (var_id, elem_ty, _) = *self
                    .field_vars
                    .get(field)
                    .ok_or_else(|| format!("field {} not declared", field))?;

                let result_ty = self.scalar_type_id(*ty);
                let alignment = Self::scalar_byte_size(*ty);

                if index.0 == u32::MAX {
                    // Push constant: access member 0 of the struct
                    let zero = self.emit_constant_u32(0);
                    let sc = if self.is_push_constant_field(*field) {
                        STORAGE_CLASS_PUSH_CONSTANT
                    } else {
                        STORAGE_CLASS_STORAGE_BUFFER
                    };
                    let ptr_elem = self.ensure_type_pointer(sc, elem_ty);
                    let chain = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_ACCESS_CHAIN,
                        &[ptr_elem, chain, var_id, zero],
                    );
                    let loaded = self.alloc_id();
                    // Memory operand 0x2 = Aligned, followed by alignment value
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_LOAD,
                        &[result_ty, loaded, chain, 0x2, alignment],
                    );
                    self.set_reg(*dst, loaded, result_ty);
                } else {
                    // Array access: struct member 0, then index into runtime array
                    let idx = self.reg_value_id(*index)?;
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
                    self.set_reg(*dst, loaded, result_ty);
                }
            }

            KernelOp::Store {
                field,
                index,
                src,
                ty,
            } => {
                let (var_id, elem_ty, _) = *self
                    .field_vars
                    .get(field)
                    .ok_or_else(|| format!("field {} not declared", field))?;

                let idx = self.reg_value_id(*index)?;
                let val = self.reg_value_id(*src)?;
                let zero = self.emit_constant_u32(0);
                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, zero, idx],
                );
                let alignment = Self::scalar_byte_size(*ty);
                // Memory operand 0x2 = Aligned, followed by alignment value
                Self::emit_op(
                    &mut self.sec_function,
                    OP_STORE,
                    &[chain, val, 0x2, alignment],
                );
            }

            KernelOp::BinOp { dst, a, b, op, ty } => {
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let result_ty = self.scalar_type_id(*ty);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );

                // Rotate ops: SPIR-V has no native rotate. Emit the
                // manual decomposition `(a << k) | (a >> (W - k))`
                // with k masked to [0, W). Falls out of the standard
                // shift + or + and + sub primitives.
                if matches!(op, BinOp::Rotl | BinOp::Rotr) {
                    let width: u32 = match ty {
                        ScalarType::U8 | ScalarType::I8 => 8,
                        ScalarType::U16 | ScalarType::I16 | ScalarType::F16 => 16,
                        ScalarType::U32 | ScalarType::I32 | ScalarType::F32 => 32,
                        ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => 64,
                        ScalarType::Bool => 1,
                    };
                    let mask = width - 1;
                    // Width/mask constants feed OpISub and OpBitwiseAnd
                    // whose result is `result_ty`, so they must share its
                    // bit width — 64-bit constants for i64/u64 rotations,
                    // 32-bit otherwise. A width mismatch is invalid SPIR-V.
                    let (mask_val, width_val) = match ty {
                        ScalarType::U64 => (
                            self.emit_constant_u64(mask as u64),
                            self.emit_constant_u64(width as u64),
                        ),
                        ScalarType::I64 => (
                            self.emit_constant_i64(mask as i64),
                            self.emit_constant_i64(width as i64),
                        ),
                        _ => (self.emit_constant_u32(mask), self.emit_constant_u32(width)),
                    };
                    // k_masked = b & mask
                    let k_masked = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_BITWISE_AND,
                        &[result_ty, k_masked, b_val, mask_val],
                    );
                    // Determine left/right shift amounts.
                    let (shl_amt, shr_amt) = if matches!(op, BinOp::Rotl) {
                        // Rotl: shl by k_masked, shr by (width - k_masked) & mask
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
                        (k_masked, w_minus_k_masked)
                    } else {
                        // Rotr: shr by k_masked, shl by (width - k_masked) & mask
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
                    self.set_reg(*dst, result, result_ty);
                    return Ok(());
                }

                // Unsigned saturating add/sub: SPIR-V has no native op.
                // The op-matrix only generates these on unsigned types.
                //   satadd(a,b) = let s = a+b in (s < a) ? MAX : s   (overflow)
                //   satsub(a,b) = (a < b) ? 0 : a-b                  (underflow)
                if matches!(op, BinOp::SatAdd | BinOp::SatSub) && !is_float && !is_signed {
                    let result = if matches!(op, BinOp::SatAdd) {
                        let sum = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_IADD,
                            &[result_ty, sum, a_val, b_val],
                        );
                        // overflow ⇔ sum < a (unsigned)
                        let bool_ty = self.ensure_type_bool();
                        let overflowed = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_ULESS_THAN,
                            &[bool_ty, overflowed, sum, a_val],
                        );
                        let max_val = self.emit_constant_unsigned_max(*ty);
                        let res = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_SELECT,
                            &[result_ty, res, overflowed, max_val, sum],
                        );
                        res
                    } else {
                        let diff = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_ISUB,
                            &[result_ty, diff, a_val, b_val],
                        );
                        // underflow ⇔ a < b (unsigned)
                        let bool_ty = self.ensure_type_bool();
                        let underflowed = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_ULESS_THAN,
                            &[bool_ty, underflowed, a_val, b_val],
                        );
                        let zero = self.emit_constant_typed_zero(*ty);
                        let res = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_SELECT,
                            &[result_ty, res, underflowed, zero, diff],
                        );
                        res
                    };
                    self.set_reg(*dst, result, result_ty);
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
                    // Unsigned sat add/sub handled by the early-return above.
                    // Float/signed sat fall back to wrapping add/sub (not
                    // generated by the matrix; no saturation modeled here).
                    (BinOp::SatAdd, true, _) => OP_FADD,
                    (BinOp::SatAdd, false, _) => OP_IADD,
                    (BinOp::SatSub, true, _) => OP_FSUB,
                    (BinOp::SatSub, false, _) => OP_ISUB,
                    // Rotates handled by the early-return above.
                    (BinOp::Rotl, _, _) | (BinOp::Rotr, _, _) => unreachable!(),
                };

                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, a_val, b_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::UnaryOp { dst, a, op, ty } => {
                let a_val = self.reg_value_id(*a)?;
                let result_ty = self.scalar_type_id(*ty);
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
                        self.set_reg(*dst, result, bool_ty);
                        return Ok(());
                    }
                }
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::Cmp { dst, a, b, op, ty } => {
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let bool_ty = self.ensure_type_bool();
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );

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
                self.set_reg(*dst, result, bool_ty);
            }

            KernelOp::Cast { dst, src, from, to } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*to);

                // Bool→int: a boolean has no bit representation in SPIR-V,
                // so OpBitcast is illegal here (it produced an invalid
                // module that drivers reject at pipeline creation with
                // VK_ERROR_UNKNOWN). Materialize 0/1 of the target type
                // with OpSelect instead. This is the Cmp→Cast(Bool,U32)
                // path that every comparison kernel takes.
                if matches!(from, ScalarType::Bool)
                    && matches!(
                        to,
                        ScalarType::U8
                            | ScalarType::U16
                            | ScalarType::U32
                            | ScalarType::U64
                            | ScalarType::I8
                            | ScalarType::I16
                            | ScalarType::I32
                            | ScalarType::I64
                    )
                {
                    let one = self.emit_constant_typed_one(*to);
                    let zero = self.emit_constant_typed_zero(*to);
                    let result = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_SELECT,
                        &[result_ty, result, src_val, one, zero],
                    );
                    self.set_reg(*dst, result, result_ty);
                    return Ok(());
                }

                let from_float =
                    matches!(from, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
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
                    _ => OP_BITCAST, // int<->int, float<->float of same size
                };
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::Copy { dst, src, ty } => {
                // In SSA, Copy is just an alias
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                self.set_reg(*dst, src_val, result_ty);
            }

            KernelOp::Branch {
                cond,
                then_ops,
                else_ops,
            } => {
                let cond_val = self.reg_value_id(*cond)?;
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
                    // No else branch — branch to then or merge
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
            }

            KernelOp::Loop {
                count,
                iter_reg,
                body,
            } => {
                let count_val = self.reg_value_id(*count)?;
                let uint_ty = self.ensure_type_u32();

                // Detect loop-carried registers: defined before the loop AND
                // written inside the body (as dst).
                let written_in_body = Self::collect_dsts(body);
                let mut carried: Vec<(u32, u32, u32)> = Vec::new(); // (reg_num, pre_loop_id, type_id)
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
                self.set_reg(*iter_reg, phi_id, uint_ty);

                // OpPhi for each loop-carried variable.
                let mut carried_phis: Vec<(u32, u32, u32, u32)> = Vec::new();
                // (reg_num, header_phi_id, continue_copy_id, type_id)
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
                    // Update register to point to the phi
                    self.set_reg(Reg(reg_num), header_phi, ty_id);
                    carried_phis.push((reg_num, header_phi, continue_copy, ty_id));
                }

                // OpLoopMerge (must be penultimate).
                //
                // Closes T1405: when `count` was defined by a Const op
                // with a small positive value (1..=8), emit
                // LOOP_CONTROL_UNROLL so the SPIR-V consumer can fully
                // unroll the loop. For larger or unknown counts, fall
                // back to LOOP_CONTROL_NONE (consumer decides). Const
                // tracking happens at the Const op emit site above via
                // `reg_const_int` — only U32/U64/I32/I64 are tracked.
                let loop_control = match self.lookup_reg_const_int(*count) {
                    Some(v) if (1..=8).contains(&v) => LOOP_CONTROL_UNROLL,
                    _ => LOOP_CONTROL_NONE,
                };
                Self::emit_op(
                    &mut self.sec_function,
                    OP_LOOP_MERGE,
                    &[merge_label, continue_label, loop_control],
                );

                // OpBranch to condition block (must be last)
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
                    // The body may have updated reg_num. Read its current value
                    // and emit an OpCopyObject so the continue_copy ID is defined.
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

                // Merge block: carried variables now hold the final value
                // from the last iteration (via the header phi).
                Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);

                // After the loop, registers should point to the header phi values
                // (which are the values from the last iteration when the loop exits).
                for &(reg_num, header_phi, _, ty_id) in &carried_phis {
                    self.set_reg(Reg(reg_num), header_phi, ty_id);
                }
            }

            KernelOp::Barrier => {
                // OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory
                let scope_wg = self.emit_constant_u32(SCOPE_WORKGROUP);
                let semantics =
                    self.emit_constant_u32(MEMORY_SEMANTICS_ACQ_REL | MEMORY_SEMANTICS_WORKGROUP);
                Self::emit_op(
                    &mut self.sec_function,
                    OP_CONTROL_BARRIER,
                    &[scope_wg, scope_wg, semantics],
                );
            }

            // OpMemoryBarrier <scope> <semantics>. We pick Workgroup scope
            // and OR the ordering bits with UniformMemory so the fence
            // applies to storage buffers (which is the point of an
            // explicit fence; thread-local memory needs no fence).
            KernelOp::Fence { order } => {
                let order_bits: u32 = match order {
                    crate::MemoryOrder::Relaxed => 0,
                    crate::MemoryOrder::Acquire => MEMORY_SEMANTICS_ACQUIRE,
                    crate::MemoryOrder::Release => MEMORY_SEMANTICS_RELEASE,
                    crate::MemoryOrder::AcqRel => MEMORY_SEMANTICS_ACQ_REL,
                    crate::MemoryOrder::SeqCst => MEMORY_SEMANTICS_SEQ_CST,
                };
                let scope_wg = self.emit_constant_u32(SCOPE_WORKGROUP);
                let semantics = self.emit_constant_u32(
                    order_bits | MEMORY_SEMANTICS_UNIFORM_MEMORY | MEMORY_SEMANTICS_WORKGROUP,
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_MEMORY_BARRIER,
                    &[scope_wg, semantics],
                );
            }

            KernelOp::SharedDecl { .. } => {
                // Already handled in emit_shared_decls
            }

            KernelOp::SharedLoad { dst, id, index, ty } => {
                let (var_id, elem_ty) = *self
                    .shared_vars
                    .get(id)
                    .ok_or_else(|| format!("shared memory {} not declared", id))?;
                let idx = self.reg_value_id(*index)?;
                let result_ty = self.scalar_type_id(*ty);

                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, idx],
                );
                let loaded = self.alloc_id();
                Self::emit_op(&mut self.sec_function, OP_LOAD, &[result_ty, loaded, chain]);
                self.set_reg(*dst, loaded, result_ty);
            }

            KernelOp::SharedStore { id, index, src, .. } => {
                let (var_id, elem_ty) = *self
                    .shared_vars
                    .get(id)
                    .ok_or_else(|| format!("shared memory {} not declared", id))?;
                let idx = self.reg_value_id(*index)?;
                let val = self.reg_value_id(*src)?;

                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, idx],
                );
                Self::emit_op(&mut self.sec_function, OP_STORE, &[chain, val]);
            }

            KernelOp::MathCall {
                dst,
                func,
                args,
                ty,
            } => {
                let ext_id = self.ensure_glsl_ext();
                let result_ty = self.scalar_type_id(*ty);
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
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::Break => {
                // Branch to the current loop's merge block.
                if let Some(&merge_label) = self.loop_merge_stack.last() {
                    Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);
                    // After a break, we need a new label for any following ops
                    // (SPIR-V requires every instruction to be in a block).
                    let dead_label = self.alloc_id();
                    Self::emit_op(&mut self.sec_function, OP_LABEL, &[dead_label]);
                } else {
                    return Err("Break outside of loop context".to_string());
                }
            }

            KernelOp::VecConstruct {
                dst,
                components,
                ty,
            } => {
                let elem_ty = self.scalar_type_id(*ty);
                let n = components.len() as u32;
                let vec_ty = self.ensure_type_vector(elem_ty, n);
                let mut ids = Vec::with_capacity(components.len());
                for c in components {
                    ids.push(self.reg_value_id(*c)?);
                }
                let result = self.alloc_id();
                let mut ops = vec![vec_ty, result];
                ops.extend_from_slice(&ids);
                Self::emit_op(&mut self.sec_function, OP_COMPOSITE_CONSTRUCT, &ops);
                self.set_reg(*dst, result, vec_ty);
            }

            KernelOp::VecExtract {
                dst,
                vec,
                component,
                ty,
            } => {
                let vec_val = self.reg_value_id(*vec)?;
                let result_ty = self.scalar_type_id(*ty);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[result_ty, result, vec_val, *component as u32],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::MatMul { dst, a, b, ty, .. } => {
                // For simple cases, matrix multiply is just component-wise
                // or dot product. SPIR-V doesn't have a generic MatMul opcode
                // for scalars. For now, treat as multiply.
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let result_ty = self.scalar_type_id(*ty);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let opcode = if is_float { OP_FMUL } else { OP_IMUL };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, a_val, b_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::DeviceCall {
                dst,
                func_name,
                args,
                ty,
            } => {
                if let Some((fn_id, ret_ty, _param_tys)) =
                    self.device_fn_ids.get(func_name).cloned()
                {
                    // Emit OpFunctionCall with the function ID and argument values
                    let result = self.alloc_id();
                    let mut operands = vec![ret_ty, result, fn_id];
                    for arg in args {
                        let arg_id = self.reg_value_id(*arg)?;
                        operands.push(arg_id);
                    }
                    Self::emit_op(&mut self.sec_function, OP_FUNCTION_CALL, &operands);
                    self.set_reg(*dst, result, ret_ty);
                } else {
                    // Fallback: emit zero constant if function not found
                    let result_ty = self.scalar_type_id(*ty);
                    let zero = match ty {
                        ScalarType::F32 | ScalarType::F16 => self.emit_constant_f32(0.0),
                        ScalarType::F64 => self.emit_constant_f64(0.0),
                        ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 => {
                            self.emit_constant_i32(0)
                        }
                        _ => self.emit_constant_u32(0),
                    };
                    self.set_reg(*dst, zero, result_ty);
                }
            }

            KernelOp::AtomicOp {
                dst,
                field,
                index,
                val,
                op,
                ty,
                order,
            } => {
                // Real SPIR-V atomic instructions (OpAtomicIAdd, etc.)
                let (var_id, elem_ty, _) = *self
                    .field_vars
                    .get(field)
                    .ok_or_else(|| format!("field {} not declared", field))?;
                let idx = self.reg_value_id(*index)?;
                let val_id = self.reg_value_id(*val)?;
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_u32(0);
                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, zero, idx],
                );

                // Scope: Device (1). Semantics derived from `order`:
                //   Relaxed → 0, Acquire → ACQUIRE, Release → RELEASE,
                //   AcqRel → ACQ_REL, SeqCst → SEQ_CST. We OR with
                //   WorkgroupMemory so the atomic also synchronizes shared
                //   memory writes — matches the prior behavior.
                let scope = self.emit_constant_u32(1);
                let order_bits: u32 = match order {
                    crate::MemoryOrder::Relaxed => 0,
                    crate::MemoryOrder::Acquire => MEMORY_SEMANTICS_ACQUIRE,
                    crate::MemoryOrder::Release => MEMORY_SEMANTICS_RELEASE,
                    crate::MemoryOrder::AcqRel => MEMORY_SEMANTICS_ACQ_REL,
                    crate::MemoryOrder::SeqCst => MEMORY_SEMANTICS_SEQ_CST,
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
                    // OpAtomicCompareExchange: result_type, result, pointer, scope, equal_sem, unequal_sem, value, comparator
                    Self::emit_op(
                        &mut self.sec_function,
                        atomic_opcode,
                        &[
                            result_ty, result_id, chain, scope, semantics, semantics, val_id,
                            val_id,
                        ],
                    );
                } else {
                    // OpAtomicIAdd etc: result_type, result, pointer, scope, semantics, value
                    Self::emit_op(
                        &mut self.sec_function,
                        atomic_opcode,
                        &[result_ty, result_id, chain, scope, semantics, val_id],
                    );
                }
                self.set_reg(*dst, result_id, result_ty);
            }

            KernelOp::AtomicCas {
                dst,
                field,
                index,
                expected,
                desired,
                ty,
                success_order,
                failure_order: _,
            } => {
                let (var_id, elem_ty, _) = *self
                    .field_vars
                    .get(field)
                    .ok_or_else(|| format!("field {} not declared", field))?;
                let idx = self.reg_value_id(*index)?;
                let exp_val = self.reg_value_id(*expected)?;
                let des_val = self.reg_value_id(*desired)?;
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_u32(0);
                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, zero, idx],
                );

                let scope = self.emit_constant_u32(1); // Device
                // SPIR-V `OpAtomicCompareExchange` takes `Equal` and
                // `Unequal` semantics; we use `success_order` for both
                // since `failure ≤ success` and SPIR-V doesn't enforce
                // the LLVM split.
                let order_bits: u32 = match success_order {
                    crate::MemoryOrder::Relaxed => 0,
                    crate::MemoryOrder::Acquire => MEMORY_SEMANTICS_ACQUIRE,
                    crate::MemoryOrder::Release => MEMORY_SEMANTICS_RELEASE,
                    crate::MemoryOrder::AcqRel => MEMORY_SEMANTICS_ACQ_REL,
                    crate::MemoryOrder::SeqCst => MEMORY_SEMANTICS_SEQ_CST,
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
                self.set_reg(*dst, result_id, result_ty);
            }

            // Shared-memory atomic: same opcode family as the buffer
            // AtomicOp above, but the pointer comes from the
            // Workgroup-storage shared variable (plain array — no
            // struct wrapper, so no leading zero index in the access
            // chain) and the Scope operand is Workgroup (2), not
            // Device (1). SPIR-V atomics don't care about the
            // storage class beyond the pointer type matching.
            KernelOp::SharedAtomicOp {
                dst,
                slot,
                index,
                val,
                op,
                ty,
                order,
            } => {
                let (var_id, elem_ty) = *self
                    .shared_vars
                    .get(slot)
                    .ok_or_else(|| format!("shared memory {} not declared", slot))?;
                let idx = self.reg_value_id(*index)?;
                let val_id = self.reg_value_id(*val)?;
                let result_ty = self.scalar_type_id(*ty);

                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, idx],
                );

                // Scope: Workgroup (2) — the contenders are the
                // lanes of this workgroup only. Semantics: order
                // bits | WorkgroupMemory, mirroring the buffer arm.
                let scope = self.emit_constant_u32(2);
                let order_bits: u32 = match order {
                    crate::MemoryOrder::Relaxed => 0,
                    crate::MemoryOrder::Acquire => MEMORY_SEMANTICS_ACQUIRE,
                    crate::MemoryOrder::Release => MEMORY_SEMANTICS_RELEASE,
                    crate::MemoryOrder::AcqRel => MEMORY_SEMANTICS_ACQ_REL,
                    crate::MemoryOrder::SeqCst => MEMORY_SEMANTICS_SEQ_CST,
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
                            result_ty, result_id, chain, scope, semantics, semantics, val_id,
                            val_id,
                        ],
                    );
                } else {
                    Self::emit_op(
                        &mut self.sec_function,
                        atomic_opcode,
                        &[result_ty, result_id, chain, scope, semantics, val_id],
                    );
                }
                self.set_reg(*dst, result_id, result_ty);
            }

            KernelOp::WaveShuffle {
                dst,
                src,
                lane_delta,
                ty,
            } => {
                // `shuffle(v, delta)` reads the value held by lane
                // `self ^ delta` — OpGroupNonUniformShuffleXor over the
                // Subgroup scope. Matches the WGSL/MSL emitters'
                // subgroupShuffleXor / simd_shuffle_xor lowering.
                let src_val = self.reg_value_id(*src)?;
                let mask_val = self.reg_value_id(*lane_delta)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_GROUP_NON_UNIFORM_SHUFFLE_XOR,
                    &[result_ty, result, scope, src_val, mask_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::WaveBallot { dst, .. }
            | KernelOp::WaveAny { dst, .. }
            | KernelOp::WaveAll { dst, .. } => {
                let uint_ty = self.ensure_type_u32();
                let zero = self.emit_constant_u32(0);
                self.set_reg(*dst, zero, uint_ty);
            }

            KernelOp::TextureSample2D { dst, ty, .. }
            | KernelOp::TextureSample3D { dst, ty, .. } => {
                // Texture sampling not yet supported
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_f32(0.0);
                self.set_reg(*dst, zero, result_ty);
            }

            KernelOp::TextureWrite2D { .. } => {
                // Texture writes not yet supported
            }

            KernelOp::TextureSize { dst_w, dst_h, .. } => {
                let uint_ty = self.ensure_type_u32();
                let zero = self.emit_constant_u32(0);
                self.set_reg(*dst_w, zero, uint_ty);
                self.set_reg(*dst_h, zero, uint_ty);
            }

            KernelOp::Bitcast { dst, src, to, .. } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*to);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_BITCAST,
                    &[result_ty, result, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::CountTrailingZeros { dst, src, ty } => {
                let ext_id = self.ensure_glsl_ext();
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_EXT_INST,
                    &[result_ty, result, ext_id, GLSL_FIND_I_LSB, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::CountLeadingZeros { dst, src, ty } => {
                let ext_id = self.ensure_glsl_ext();
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let msb = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_EXT_INST,
                    &[result_ty, msb, ext_id, GLSL_FIND_U_MSB, src_val],
                );
                let thirty_one = self.emit_constant_u32(31);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ISUB,
                    &[result_ty, result, thirty_one, msb],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::PopCount { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_BIT_COUNT,
                    &[result_ty, result, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::Dot { dst, a, b, ty, .. } => {
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let result_ty = self.scalar_type_id(*ty);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_DOT,
                    &[result_ty, result, a_val, b_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupReduceAdd { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FADD
                } else {
                    OP_GROUP_NON_UNIFORM_IADD
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, scope, GROUP_OPERATION_REDUCE, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupReduceMin { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FMIN
                } else if is_signed {
                    OP_GROUP_NON_UNIFORM_SMIN
                } else {
                    OP_GROUP_NON_UNIFORM_UMIN
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, scope, GROUP_OPERATION_REDUCE, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupReduceMax { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FMAX
                } else if is_signed {
                    OP_GROUP_NON_UNIFORM_SMAX
                } else {
                    OP_GROUP_NON_UNIFORM_UMAX
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, scope, GROUP_OPERATION_REDUCE, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupExclusiveAdd { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FADD
                } else {
                    OP_GROUP_NON_UNIFORM_IADD
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[
                        result_ty,
                        result,
                        scope,
                        GROUP_OPERATION_EXCLUSIVE_SCAN,
                        src_val,
                    ],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupInclusiveAdd { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FADD
                } else {
                    OP_GROUP_NON_UNIFORM_IADD
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[
                        result_ty,
                        result,
                        scope,
                        GROUP_OPERATION_INCLUSIVE_SCAN,
                        src_val,
                    ],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::TextureLoad2D { dst, ty, .. } => {
                // Texture load not yet wired to image variables; placeholder zero.
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_f32(0.0);
                self.set_reg(*dst, zero, result_ty);
            }

            KernelOp::SubgroupSize { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.emit_constant_u32(32); // placeholder: common subgroup size
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::SharedDeclDyn { .. } => {
                // Handled during shared decl scan phase.
            }

            KernelOp::DebugPrint { src, ty } => {
                // Debug print: no-op in SPIR-V for now.
                let _ = (src, ty);
            }

            KernelOp::Dispatch { .. } => {
                // Dynamic parallelism not supported in Vulkan compute
            }

            KernelOp::CooperativeMMA { dst, ty, .. } => {
                // Cooperative matrix multiply-add not yet supported; placeholder zero.
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_f32(0.0);
                self.set_reg(*dst, zero, result_ty);
            }
        }

        Ok(())
    }
}
