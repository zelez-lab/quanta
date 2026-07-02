//! KernelOp → SPIR-V instruction dispatch.
//!
//! The single `emit_single_op` method handles every KernelOp variant,
//! emitting the corresponding SPIR-V instruction(s) into sec_function.

use crate::*;

use super::constants::*;
use super::emitter::SpvEmitter;

/// `(exp_bits, mant_bits)` for an fp8 scalar type, else `None`.
fn fp8_dims(ty: &ScalarType) -> Option<(u32, u32)> {
    match ty {
        ScalarType::FP8E5M2 => Some((5, 2)),
        ScalarType::FP8E4M3 => Some((4, 3)),
        _ => None,
    }
}

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
                // Folded-dispatch linearization: gid.x + gid.y *
                // (FOLD_ROW_GROUPS * wg_x). Identity for plain 1D
                // dispatches (gid.y == 0); exact continuation across
                // the rectangle + remainder rows of a folded one.
                let uint_ty = self.ensure_type_u32();
                let row_span = crate::dispatch_fold::FOLD_ROW_GROUPS * self.wg_x;
                let val = self.load_builtin_linear(gid_var, row_span);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::ProtonId { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.load_builtin_x(proton_id_var);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::NucleusId { dst } => {
                // Same linearization as QuarkId, at workgroup
                // granularity: wg_id.x + wg_id.y * FOLD_ROW_GROUPS.
                let uint_ty = self.ensure_type_u32();
                let val =
                    self.load_builtin_linear(nucleus_id_var, crate::dispatch_fold::FOLD_ROW_GROUPS);
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
                    // Int SSA values are canonically unsigned (see
                    // scalar_type_id): emit signed constants as their `%uint`
                    // bit pattern. Two's-complement means `v as u32` is the same
                    // bits; a later signed op bitcasts to `%int` as needed.
                    ConstValue::I32(v) => {
                        let ty = self.ensure_type_u32();
                        (self.emit_constant_u32(*v as u32), ty)
                    }
                    ConstValue::I64(v) => {
                        let ty = self.ensure_type_u64();
                        (self.emit_constant_u64(*v as u64), ty)
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
                    ConstValue::BF16(v) => {
                        // bf16 is the top 16 bits of an f32 — unpack by
                        // left-shifting into place. Emulated body is f32.
                        let ty = self.ensure_type_f32();
                        let f = f32::from_bits((*v as u32) << 16);
                        (self.emit_constant_f32(f), ty)
                    }
                    ConstValue::FP8E5M2(v) => {
                        let ty = self.ensure_type_f32();
                        let (eb, mb) = crate::dtype::E5M2;
                        (
                            self.emit_constant_f32(crate::dtype::fp8_to_f32(*v, eb, mb)),
                            ty,
                        )
                    }
                    ConstValue::FP8E4M3(v) => {
                        let ty = self.ensure_type_f32();
                        let (eb, mb) = crate::dtype::E4M3;
                        (
                            self.emit_constant_f32(crate::dtype::fp8_to_f32(*v, eb, mb)),
                            ty,
                        )
                    }
                };
                self.set_reg(*dst, id, ty);
                // Track integer constants so the Loop emitter can pick
                // LOOP_CONTROL_UNROLL for short iteration counts (TODO T1405).
                // Floats and booleans aren't tracked — they don't feed
                // Loop.count. Demoted (mutable) registers aren't tracked
                // either: their value can change after this Const.
                if self.demoted_regs.contains_key(&dst.0) {
                    return Ok(());
                }
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
                    // Push constant: access this slot's member of the
                    // shared block
                    let member_idx = self.push_constant_member.get(field).copied().unwrap_or(0);
                    let member = self.emit_constant_u32(member_idx);
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
                        &[ptr_elem, chain, var_id, member],
                    );
                    let loaded = self.alloc_id();
                    let storage_ty = self.storage_scalar_type_id(*ty);
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_LOAD,
                        &[storage_ty, loaded, chain, 0x2, alignment],
                    );
                    let val = if matches!(ty, ScalarType::BF16) {
                        self.bf16_unpack_to_f32(loaded)
                    } else if let Some((eb, mb)) = fp8_dims(ty) {
                        self.fp8_unpack_to_f32(loaded, eb, mb)
                    } else {
                        loaded
                    };
                    self.set_reg(*dst, val, result_ty);
                } else if matches!(ty, ScalarType::I4) {
                    // int4 PackedU32: word = idx/8, nibble = idx%8. Load the
                    // word, extract and sign-extend the nibble to i32.
                    let idx = self.reg_value_id(*index)?;
                    let val = self.i4_load_nibble(var_id, idx);
                    self.set_reg(*dst, val, result_ty);
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
                    // Load with the *storage* element type, then unpack to
                    // the body type for bf16 (storage u16/u32 → f32).
                    let storage_ty = self.storage_scalar_type_id(*ty);
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_LOAD,
                        &[storage_ty, loaded, chain, 0x2, alignment],
                    );
                    let val = if matches!(ty, ScalarType::BF16) {
                        self.bf16_unpack_to_f32(loaded)
                    } else if let Some((eb, mb)) = fp8_dims(ty) {
                        self.fp8_unpack_to_f32(loaded, eb, mb)
                    } else {
                        loaded
                    };
                    self.set_reg(*dst, val, result_ty);
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
                let mut val = self.reg_value_id(*src)?;
                // A bool value stored into a numeric buffer (e.g.
                // `out[i] = (a < b) as u32`) must be materialized as an int —
                // OpStore is strictly typed and there's no bool element type.
                let bool_ty = self.ensure_type_bool();
                let src_ty = self.reg_type_id(*src)?;
                if src_ty == bool_ty && elem_ty != bool_ty {
                    val = self.coerce_to(val, bool_ty, elem_ty);
                }
                // int4 PackedU32: write nibble idx%8 of word idx/8 via
                // read-modify-write (single-quark in the op-matrix).
                if matches!(ty, ScalarType::I4) {
                    self.i4_store_nibble(var_id, idx, val);
                    return Ok(());
                }
                // bf16: pack the f32 body value into its storage bits first.
                let stored_val = if matches!(ty, ScalarType::BF16) {
                    self.bf16_pack_from_f32(val)
                } else if let Some((eb, mb)) = fp8_dims(ty) {
                    self.fp8_pack_from_f32(val, eb, mb)
                } else {
                    val
                };
                let zero = self.emit_constant_u32(0);
                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, zero, idx],
                );
                let alignment = self.storage_byte_size(*ty);
                // Memory operand 0x2 = Aligned, followed by alignment value
                Self::emit_op(
                    &mut self.sec_function,
                    OP_STORE,
                    &[chain, stored_val, 0x2, alignment],
                );
            }

            KernelOp::BinOp { dst, a, b, op, ty } => {
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let a_ty = self.reg_type_id(*a)?;
                let b_ty = self.reg_type_id(*b)?;
                let result_ty = self.scalar_type_id(*ty);
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
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
                        // bf16's body is f32; rotate-on-float is never
                        // generated, but the width must be exhaustive.
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

                // Signed integer ops (SDiv/SRem/arithmetic-shift-right) need
                // `%int` operands and produce an `%int` result; every other int
                // op works on the canonical `%uint`. Since int SSA values are
                // canonically `%uint`, bitcast operands to the op's expected type
                // and the result back to `%uint`. Floats use their own type.
                let needs_signed = matches!(
                    (op, is_signed),
                    (BinOp::Div, true) | (BinOp::Rem, true) | (BinOp::Shr, true)
                );
                let op_ty = if is_float {
                    result_ty
                } else if needs_signed {
                    self.ensure_type_i32_for(*ty)
                } else {
                    result_ty
                };
                let a_val = self.coerce_to(a_val, a_ty, op_ty);
                let b_val = self.coerce_to(b_val, b_ty, op_ty);
                let raw = self.alloc_id();
                Self::emit_op(&mut self.sec_function, opcode, &[op_ty, raw, a_val, b_val]);
                let result = self.coerce_to(raw, op_ty, result_ty);
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::UnaryOp { dst, a, op, ty } => {
                let a_val = self.reg_value_id(*a)?;
                let result_ty = self.scalar_type_id(*ty);
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );

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
                let a_ty = self.reg_type_id(*a)?;
                let b_ty = self.reg_type_id(*b)?;
                let bool_ty = self.ensure_type_bool();
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
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

                // Operands must share the type the opcode expects: `%int` for a
                // signed compare (S*), the canonical `%uint` for unsigned/float.
                // Coerce any int operand that arrived as the other signedness.
                let operand_ty = if is_float {
                    self.scalar_type_id(*ty)
                } else if is_signed {
                    self.ensure_type_i32_for(*ty)
                } else if matches!(ty, ScalarType::Bool) {
                    // OpIEqual & friends take *int* operands — a Bool-typed
                    // compare lane compares the 0/1 materializations
                    // (coerce_to turns a %bool operand into OpSelect 1/0),
                    // never `%bool` values directly.
                    self.ensure_type_u32()
                } else {
                    self.scalar_type_id(*ty)
                };
                let a_val = self.coerce_to(a_val, a_ty, operand_ty);
                let b_val = self.coerce_to(b_val, b_ty, operand_ty);
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

                // Int→bool: OpBitcast to %bool is equally illegal (bools
                // have no bit representation). Truthiness-test instead —
                // `coerce_to` emits `OpINotEqual src, 0` for int sources.
                if matches!(to, ScalarType::Bool) && !matches!(from, ScalarType::Bool) {
                    let src_ty = self.reg_type_id(*src)?;
                    let bool_ty = self.ensure_type_bool();
                    let result = self.coerce_to(src_val, src_ty, bool_ty);
                    self.set_reg(*dst, result, bool_ty);
                    return Ok(());
                }

                let from_float = matches!(
                    from,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
                let to_float = matches!(
                    to,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
                let from_signed = matches!(
                    from,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );
                let to_signed = matches!(
                    to,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );

                // Same canonical SPIR-V type (e.g. u32↔i32 — both `%uint`):
                // a pure alias. OpBitcast between identical types is invalid
                // SPIR-V.
                let src_ty = self.reg_type_id(*src)?;
                if src_ty == result_ty {
                    self.set_reg(*dst, src_val, result_ty);
                    return Ok(());
                }

                let opcode = match (from_float, to_float, from_signed, to_signed) {
                    (false, true, true, _) => OP_CONVERT_S_TO_F,
                    (false, true, false, _) => OP_CONVERT_U_TO_F,
                    (true, false, _, true) => OP_CONVERT_F_TO_S,
                    (true, false, _, false) => OP_CONVERT_F_TO_U,
                    // Float width change (f32↔f64, f16↔f32): a real
                    // conversion — OpBitcast across widths is invalid.
                    (true, true, _, _) => OP_F_CONVERT,
                    (false, false, _, _) => {
                        // Int↔int. Same canonical width is a free
                        // reinterpret; a width change needs a real
                        // conversion (OpBitcast can't change total width —
                        // it produced invalid modules for the u64 pack
                        // shape `(hi as u64) << 32 | lo`).
                        let from_64 =
                            self.type_u64 == Some(src_ty) || self.type_i64 == Some(src_ty);
                        let to_64 =
                            self.type_u64 == Some(result_ty) || self.type_i64 == Some(result_ty);
                        if from_64 == to_64 {
                            OP_BITCAST
                        } else if to_64 && from_signed {
                            // Sign-extend: bitcast the canonical `%uint` to
                            // `%int`, OpSConvert to `%long`, bitcast back to
                            // the canonical `%ulong`.
                            let int_ty = self.ensure_type_i32();
                            let long_ty = self.ensure_type_i64();
                            let s = self.coerce_to(src_val, src_ty, int_ty);
                            let ext = self.alloc_id();
                            Self::emit_op(&mut self.sec_function, OP_S_CONVERT, &[long_ty, ext, s]);
                            let out = self.coerce_to(ext, long_ty, result_ty);
                            self.set_reg(*dst, out, result_ty);
                            return Ok(());
                        } else {
                            // Zero-extend (32→64 unsigned) or truncate (64→32).
                            OP_U_CONVERT
                        }
                    }
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::Copy { dst, src, ty } => {
                // For a demoted (mutable) dst this materializes as an
                // OpStore into its variable via set_reg — the anchoring
                // Copies the lowering emits for loop-carried / branch-
                // assigned locals MUST produce a real write. A single-def
                // dst stays a pure SSA alias.
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                self.set_reg(*dst, src_val, result_ty);
            }

            // Per-tensor symmetric quantize:
            //   q = clamp(i32(RoundEven(x / scale)), lo, hi)
            KernelOp::Quantize {
                dst,
                src,
                scale,
                scheme,
                ..
            } => {
                let (lo, hi) = scheme.value.range();
                let x = self.reg_value_id(*src)?;
                let s = self.reg_value_id(*scale)?;
                let f32_ty = self.ensure_type_f32();
                let i32_ty = self.ensure_type_i32();
                let ext = self.ensure_glsl_ext();
                // x / scale
                let div = self.alloc_id();
                Self::emit_op(&mut self.sec_function, OP_FDIV, &[f32_ty, div, x, s]);
                // RoundEven
                let rounded = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_EXT_INST,
                    &[f32_ty, rounded, ext, GLSL_ROUND_EVEN, div],
                );
                // i32(rounded)
                let q = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_CONVERT_F_TO_S,
                    &[i32_ty, q, rounded],
                );
                // clamp: SMax(q, lo) then SMin(·, hi)
                let lo_c = self.emit_constant_i32(lo);
                let hi_c = self.emit_constant_i32(hi);
                let cmax = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_EXT_INST,
                    &[i32_ty, cmax, ext, GLSL_SMAX, q, lo_c],
                );
                let cmin = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_EXT_INST,
                    &[i32_ty, cmin, ext, GLSL_SMIN, cmax, hi_c],
                );
                self.set_reg(*dst, cmin, i32_ty);
            }
            // Per-tensor symmetric dequantize: dq = scale * f32(q).
            KernelOp::Dequantize {
                dst, src, scale, ..
            } => {
                let q = self.reg_value_id(*src)?;
                let s = self.reg_value_id(*scale)?;
                let f32_ty = self.ensure_type_f32();
                let qf = self.alloc_id();
                Self::emit_op(&mut self.sec_function, OP_CONVERT_S_TO_F, &[f32_ty, qf, q]);
                let dq = self.alloc_id();
                Self::emit_op(&mut self.sec_function, OP_FMUL, &[f32_ty, dq, s, qf]);
                self.set_reg(*dst, dq, f32_ty);
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
                // written inside the body (as dst). Demoted (mutable)
                // registers never appear here — they live in `demoted_regs`,
                // not `reg_ids`, and go through their Function-storage
                // variable instead of header phis.
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

                // Bind the counter register AFTER all header phis are
                // emitted: for a demoted iter_reg this set_reg emits an
                // OpStore, and no non-phi instruction may precede a phi.
                self.set_reg(*iter_reg, phi_id, uint_ty);

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
                    // If the body reassigned it to the other int signedness,
                    // coerce back to the phi's type so the copy (and the phi edge
                    // it feeds) stay type-consistent.
                    let current = self.reg_ids[&reg_num];
                    let current_ty = self.reg_types[&reg_num];
                    let current = self.coerce_to(current, current_ty, ty_id);
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
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
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

                // The transcendental GLSL.std.450 instructions (Sin/Cos/…/
                // Exp/Log/Pow) accept only 16- or 32-bit floats. There is no
                // correct f64 path: emulating one by narrowing to f32 is
                // silently lossy (it corrupts algorithms whose argument can be
                // tiny, e.g. Box-Muller's ln()), so f64 transcendentals are
                // refused outright. `validate_for(VULKAN, …)` reports them as
                // NotSupported and callers get a clean error, never wrong
                // numbers — this is the defensive backstop for that gate.
                if matches!(ty, ScalarType::F64) && is_f64_transcendental(*func) {
                    return Err("f64 transcendental math is not supported on the SPIR-V \
                         backend (GLSL.std.450 has no f64 variant)"
                        .to_string());
                }

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
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
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
                        ScalarType::F32
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3 => self.emit_constant_f32(0.0),
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
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
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
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
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
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
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
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
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
                let is_float = matches!(
                    ty,
                    ScalarType::F32
                        | ScalarType::F64
                        | ScalarType::F16
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                );
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
            KernelOp::CooperativeMatrixLoad { dst, ty, .. } => {
                // SPV_KHR_cooperative_matrix codegen is a later arm (Metal-first);
                // placeholder zero until then.
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_f32(0.0);
                self.set_reg(*dst, zero, result_ty);
            }
            KernelOp::CooperativeMatrixStore { .. } => {
                // Placeholder no-op until native cooperative-matrix store lands.
            }
        }

        Ok(())
    }

    // ── bf16 storage conversions ─────────────────────────────────────────
    //
    // bf16 is stored as a 16-bit pattern (the top half of an f32) but
    // computed in f32. These convert at the Load/Store boundary. The
    // packing rounds to nearest-even and must match the CPU executor's
    // `f32_to_bf16` bit-for-bit (the differential oracle).

    /// Convert a loaded bf16 storage value (`u16` native, or `u32` carrying
    /// the bits in its low half) into an f32 register: `f32 = bitcast(bits
    /// << 16)`. Returns the f32 value id.
    pub(crate) fn bf16_unpack_to_f32(&mut self, loaded: u32) -> u32 {
        let u32_ty = self.ensure_type_u32();
        let f32_ty = self.ensure_type_f32();
        // Widen to u32 if the storage element was u16 (native path).
        let bits32 = if self.caps.bf16_native_storage {
            let w = self.alloc_id();
            Self::emit_op(&mut self.sec_function, OP_U_CONVERT, &[u32_ty, w, loaded]);
            w
        } else {
            loaded
        };
        let sixteen = self.emit_constant_u32(16);
        let shifted = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_SHIFT_LEFT_LOGICAL,
            &[u32_ty, shifted, bits32, sixteen],
        );
        let f = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[f32_ty, f, shifted]);
        f
    }

    /// Pack an f32 register into a bf16 storage value (`u16` native / `u32`
    /// fallback), round-to-nearest-even:
    ///   bits = bitcast<u32>(f); bias = 0x7fff + ((bits >> 16) & 1);
    ///   out  = (bits + bias) >> 16
    /// (NaN handling: a NaN's exponent is all-ones so the bias never
    /// overflows it into ±inf, matching the CPU path for finite values; the
    /// op-matrix oracle uses the same formula and skips NaN cases.)
    /// Returns the storage value id (u16 or u32 per caps).
    pub(crate) fn bf16_pack_from_f32(&mut self, f32_val: u32) -> u32 {
        let u32_ty = self.ensure_type_u32();
        let bits = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[u32_ty, bits, f32_val]);
        // lsb = (bits >> 16) & 1
        let sixteen = self.emit_constant_u32(16);
        let one = self.emit_constant_u32(1);
        let hi = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_SHIFT_RIGHT_LOGICAL,
            &[u32_ty, hi, bits, sixteen],
        );
        let lsb = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_BITWISE_AND,
            &[u32_ty, lsb, hi, one],
        );
        // bias = 0x7fff + lsb
        let base_bias = self.emit_constant_u32(0x7fff);
        let bias = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_IADD,
            &[u32_ty, bias, base_bias, lsb],
        );
        // rounded = (bits + bias) >> 16
        let summed = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_IADD,
            &[u32_ty, summed, bits, bias],
        );
        let rounded = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_SHIFT_RIGHT_LOGICAL,
            &[u32_ty, rounded, summed, sixteen],
        );
        // Narrow to u16 for the native storage element.
        if self.caps.bf16_native_storage {
            let u16_ty = self.ensure_type_u16();
            let narrowed = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_U_CONVERT,
                &[u16_ty, narrowed, rounded],
            );
            narrowed
        } else {
            rounded
        }
    }

    // ── fp8 storage conversions ──────────────────────────────────────────
    //
    // The branchless reference is `crate::dtype::{fp8_to_f32, f32_to_fp8}`;
    // these emit the identical arithmetic op-by-op. Storage is the portable
    // u32-slot (one fp8 byte per word). `eb`/`mb` are the exponent/mantissa
    // widths. Helpers below are thin one-op wrappers so the conversion reads
    // close to the Rust source.

    fn spv_u32(&mut self) -> u32 {
        self.ensure_type_u32()
    }
    fn spv_const(&mut self, v: u32) -> u32 {
        self.emit_constant_u32(v)
    }
    /// Emit a 2-arg op producing `ty`, return the result id.
    fn spv_bin(&mut self, opcode: u16, ty: u32, a: u32, b: u32) -> u32 {
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_function, opcode, &[ty, id, a, b]);
        id
    }
    fn spv_and(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_BITWISE_AND, t, a, b)
    }
    fn spv_or(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_BITWISE_OR, t, a, b)
    }
    fn spv_add(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_IADD, t, a, b)
    }
    fn spv_sub(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_ISUB, t, a, b)
    }
    fn spv_mul(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_IMUL, t, a, b)
    }
    fn spv_shl(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_SHIFT_LEFT_LOGICAL, t, a, b)
    }
    fn spv_shr(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_SHIFT_RIGHT_LOGICAL, t, a, b)
    }
    /// A bool-typed comparison.
    fn spv_cmp(&mut self, opcode: u16, a: u32, b: u32) -> u32 {
        let bt = self.ensure_type_bool();
        self.spv_bin(opcode, bt, a, b)
    }
    fn spv_logic(&mut self, opcode: u16, a: u32, b: u32) -> u32 {
        let bt = self.ensure_type_bool();
        self.spv_bin(opcode, bt, a, b)
    }
    /// `select(cond ? a : b)` over u32 (SPIR-V operand order is cond,a,b).
    fn spv_select(&mut self, ty: u32, cond: u32, a: u32, b: u32) -> u32 {
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_SELECT, &[ty, id, cond, a, b]);
        id
    }

    /// Round-to-nearest-even right shift by a constant `s` (mirrors
    /// `dtype::round_shift_rne` for the constant-shift cases used here:
    /// `s` is always a known small value, so no runtime clamp is needed).
    fn spv_rne_const(&mut self, v: u32, s: u32) -> u32 {
        if s == 0 {
            return v;
        }
        let u = self.spv_u32();
        let s_c = self.spv_const(s);
        let kept = self.spv_shr(v, s_c);
        let mask = self.spv_const((1u32 << s) - 1);
        let rem = self.spv_and(v, mask);
        let half = self.spv_const(1u32 << (s - 1));
        // roundup = rem > half || (rem == half && (kept&1)==1)
        let gt = self.spv_cmp(OP_UGREATER_THAN, rem, half);
        let eq = self.spv_cmp(OP_IEQUAL, rem, half);
        let one = self.spv_const(1);
        let kept_lsb = self.spv_and(kept, one);
        let lsb_set = self.spv_cmp(OP_IEQUAL, kept_lsb, one);
        let tie_up = self.spv_logic(OP_LOGICAL_AND, eq, lsb_set);
        let roundup = self.spv_logic(OP_LOGICAL_OR, gt, tie_up);
        let zero = self.spv_const(0);
        let inc = self.spv_select(u, roundup, one, zero);
        self.spv_add(kept, inc)
    }

    /// Unpack a loaded fp8 u32-slot value into an f32 register.
    /// Mirrors `dtype::fp8_to_f32`.
    pub(crate) fn fp8_unpack_to_f32(&mut self, loaded: u32, eb: u32, mb: u32) -> u32 {
        let u = self.spv_u32();
        let f32_ty = self.ensure_type_f32();
        let bias = (1u32 << (eb - 1)) - 1;
        let exp_mask = (1u32 << eb) - 1;
        let mant_mask = (1u32 << mb) - 1;

        let sm = self.spv_const(eb + mb);
        let sign = {
            let s = self.spv_shr(loaded, sm);
            let one = self.spv_const(1);
            self.spv_and(s, one)
        };
        let exp = {
            let mbc = self.spv_const(mb);
            let s = self.spv_shr(loaded, mbc);
            let m = self.spv_const(exp_mask);
            self.spv_and(s, m)
        };
        let mant = {
            let m = self.spv_const(mant_mask);
            self.spv_and(loaded, m)
        };
        let c31 = self.spv_const(31);
        let fsign = self.spv_shl(sign, c31);

        // norm = fsign | ((exp + 127 - bias) << 23) | (mant << (23-mb))
        let c127 = self.spv_const(127);
        let cbias = self.spv_const(bias);
        let e1 = self.spv_add(exp, c127);
        let e2 = self.spv_sub(e1, cbias);
        let c23 = self.spv_const(23);
        let es = self.spv_shl(e2, c23);
        let clo = self.spv_const(23 - mb);
        let ms = self.spv_shl(mant, clo);
        let norm = {
            let t = self.spv_or(fsign, es);
            self.spv_or(t, ms)
        };

        // infnan = fsign | (0xFF << 23) | (mant!=0 ? 0x400000 : 0)
        let c0xff = self.spv_const(0xFF);
        let ff_sh = self.spv_shl(c0xff, c23);
        let zero = self.spv_const(0);
        let cz = self.spv_const(0x0040_0000);
        let mant_nz = self.spv_cmp(OP_INOT_EQUAL, mant, zero);
        let inf_mant = self.spv_select(u, mant_nz, cz, zero);
        let infnan = {
            let t = self.spv_or(fsign, ff_sh);
            self.spv_or(t, inf_mant)
        };

        // subnormal: unrolled leading-bit scan over mb bits.
        let mut lead = self.spv_const(0);
        for i in 0..mb {
            let ci = self.spv_const(i);
            let one = self.spv_const(1);
            let sh = self.spv_shr(mant, ci);
            let bit = self.spv_and(sh, one);
            // lead = bit*i | (1-bit)*lead
            let bi = self.spv_mul(bit, ci);
            let inv = self.spv_sub(one, bit);
            let il = self.spv_mul(inv, lead);
            lead = self.spv_or(bi, il);
        }
        let cmb = self.spv_const(mb);
        let shifts = self.spv_sub(cmb, lead);
        // e_sub + mb + 127 = (1 - bias - mb - shifts) + mb + 127
        //                  = 128 - bias - shifts
        let c_base = self.spv_const(128 - bias);
        let e_sub_biased = self.spv_sub(c_base, shifts);
        let m_shifted = self.spv_shl(mant, shifts);
        let mm = self.spv_const(mant_mask);
        let m_sub = self.spv_and(m_shifted, mm);
        let sub = {
            let es2 = self.spv_shl(e_sub_biased, c23);
            let ms2 = self.spv_shl(m_sub, clo);
            let t = self.spv_or(fsign, es2);
            self.spv_or(t, ms2)
        };

        // predicates
        let cem = self.spv_const(exp_mask);
        let is_inf = self.spv_cmp(OP_IEQUAL, exp, cem);
        let exp_z = self.spv_cmp(OP_IEQUAL, exp, zero);
        let mant_z = self.spv_cmp(OP_IEQUAL, mant, zero);
        let is_zero = self.spv_logic(OP_LOGICAL_AND, exp_z, mant_z);
        let is_sub = self.spv_logic(OP_LOGICAL_AND, exp_z, mant_nz);

        // out = select chain: norm; sub; infnan; zero(=fsign)
        let mut out = norm;
        out = self.spv_select(u, is_sub, sub, out);
        out = self.spv_select(u, is_inf, infnan, out);
        out = self.spv_select(u, is_zero, fsign, out);

        let f = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[f32_ty, f, out]);
        f
    }

    /// Pack an f32 register into an fp8 u32-slot value. Mirrors
    /// `dtype::f32_to_fp8`.
    pub(crate) fn fp8_pack_from_f32(&mut self, f32_val: u32, eb: u32, mb: u32) -> u32 {
        let u = self.spv_u32();
        let bias = (1u32 << (eb - 1)) - 1;
        let exp_mask = (1u32 << eb) - 1;
        let mant_mask = (1u32 << mb) - 1;

        let b = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[u, b, f32_val]);

        let c31 = self.spv_const(31);
        let one = self.spv_const(1);
        let zero = self.spv_const(0);
        let sign = {
            let s = self.spv_shr(b, c31);
            self.spv_and(s, one)
        };
        let sm = self.spv_const(eb + mb);
        let sign_slot = self.spv_shl(sign, sm);
        let c23 = self.spv_const(23);
        let c0xff = self.spv_const(0xFF);
        let fexp = {
            let s = self.spv_shr(b, c23);
            self.spv_and(s, c0xff)
        };
        let cmant = self.spv_const(0x007F_FFFF);
        let fmant = self.spv_and(b, cmant);

        // target_exp = (fexp - 127) + bias = fexp + (bias - 127), as i32.
        // Work signed: use signed ops. Represent constants as u32 bit
        // patterns; SPIR-V int ops are sign-agnostic, only comparisons
        // differ. We compare with signed ops where the reference does.
        let c127 = self.spv_const(127);
        let cbias = self.spv_const(bias);
        // target_exp = fexp - 127 + bias  (may be negative → wraps in u32)
        let te0 = self.spv_sub(fexp, c127);
        let target_exp = self.spv_add(te0, cbias);

        // normal: rnd_n = rne(fmant, 23-mb)
        let rnd_n = self.spv_rne_const(fmant, 23 - mb);
        let cmb = self.spv_const(mb);
        let rnd_hi = self.spv_shr(rnd_n, cmb);
        let carry = self.spv_cmp(OP_INOT_EQUAL, rnd_hi, zero);
        let carry_u = self.spv_select(u, carry, one, zero);
        let out_exp_n = self.spv_add(target_exp, carry_u);
        let out_mant_n = self.spv_select(u, carry, zero, rnd_n);
        let cem = self.spv_const(exp_mask);
        let em_sh = self.spv_shl(cem, cmb);
        let inf_val = self.spv_or(sign_slot, em_sh);
        let normal = {
            let oe = self.spv_shl(out_exp_n, cmb);
            let mm = self.spv_const(mant_mask);
            let om = self.spv_and(out_mant_n, mm);
            let t = self.spv_or(sign_slot, oe);
            let body = self.spv_or(t, om);
            // out_exp_n >= exp_mask ? inf : body  (unsigned compare ok:
            // a wrapped-negative target_exp is huge → ≥ exp_mask → inf,
            // but that case is overridden by is_sub below anyway)
            let oexp_big = self.spv_cmp(OP_UGREATER_THAN_EQUAL, out_exp_n, cem);
            self.spv_select(u, oexp_big, inf_val, body)
        };

        // subnormal: shift = (23-mb) + (1 - target_exp)
        //   = (24 - mb) - target_exp ; signed.
        let signif = {
            let c_imp = self.spv_const(0x0080_0000);
            self.spv_or(fmant, c_imp)
        };
        let c24mb = self.spv_const(24 - mb);
        let shift = self.spv_sub(c24mb, target_exp); // signed value in u32
        // rne with a *runtime* shift; reuse the variable-shift form.
        let sub_rne = self.spv_rne_var(signif, shift);
        let sub_body = self.spv_or(sign_slot, sub_rne);
        // shift > 31 (signed) → ±0
        let c31s = self.spv_const(31);
        let shift_big = self.spv_cmp(OP_SGREATER_THAN, shift, c31s);
        let sub = self.spv_select(u, shift_big, sign_slot, sub_body);

        // inf/nan, overflow, zero
        let nan_m = {
            let c = self.spv_const(1u32 << (mb - 1));
            let nz = self.spv_cmp(OP_INOT_EQUAL, fmant, zero);
            self.spv_select(u, nz, c, zero)
        };
        let infnan = {
            let t = self.spv_or(sign_slot, em_sh);
            self.spv_or(t, nan_m)
        };
        let ovf = self.spv_or(sign_slot, em_sh);

        // predicates (signed where the reference is signed)
        let is_infnan = self.spv_cmp(OP_IEQUAL, fexp, c0xff);
        let fexp_z = self.spv_cmp(OP_IEQUAL, fexp, zero);
        let fmant_z = self.spv_cmp(OP_IEQUAL, fmant, zero);
        let is_zero = self.spv_logic(OP_LOGICAL_AND, fexp_z, fmant_z);
        let is_ovf = self.spv_cmp(OP_SGREATER_THAN_EQUAL, target_exp, cem);
        let is_sub = self.spv_cmp(OP_SLESS_THAN_EQUAL, target_exp, zero);

        // out = normal; sub; ovf; zero; infnan  (priority bottom→top)
        let mut out = normal;
        out = self.spv_select(u, is_sub, sub, out);
        out = self.spv_select(u, is_ovf, ovf, out);
        out = self.spv_select(u, is_zero, sign_slot, out);
        out = self.spv_select(u, is_infnan, infnan, out);
        out
    }

    /// Round-to-nearest-even right shift by a *runtime* shift amount,
    /// branchless over `s` (clamps `s>=32` → 0 like `dtype::round_shift_rne`).
    fn spv_rne_var(&mut self, v: u32, s: u32) -> u32 {
        let u = self.spv_u32();
        let c32 = self.spv_const(32);
        let big = self.spv_cmp(OP_UGREATER_THAN_EQUAL, s, c32);
        let c31 = self.spv_const(31);
        let s_c = self.spv_select(u, big, c31, s);
        let kept = self.spv_shr(v, s_c);
        // mask = (1 << s_c) - 1
        let one = self.spv_const(1);
        let sh1 = self.spv_shl(one, s_c);
        let mask = self.spv_sub(sh1, one);
        let rem = self.spv_and(v, mask);
        // half = 1 << (s_c==0 ? 0 : s_c-1)
        let zero = self.spv_const(0);
        let sc_z = self.spv_cmp(OP_IEQUAL, s_c, zero);
        let scm1 = self.spv_sub(s_c, one);
        let hshift = self.spv_select(u, sc_z, zero, scm1);
        let half = self.spv_shl(one, hshift);
        let gt = self.spv_cmp(OP_UGREATER_THAN, rem, half);
        let eq = self.spv_cmp(OP_IEQUAL, rem, half);
        let kept_lsb = self.spv_and(kept, one);
        let lsb_set = self.spv_cmp(OP_IEQUAL, kept_lsb, one);
        let tie_up = self.spv_logic(OP_LOGICAL_AND, eq, lsb_set);
        let roundup = self.spv_logic(OP_LOGICAL_OR, gt, tie_up);
        let inc = self.spv_select(u, roundup, one, zero);
        let r = self.spv_add(kept, inc);
        // s==0 ? v : r
        let s_z = self.spv_cmp(OP_IEQUAL, s, zero);
        let r0 = self.spv_select(u, s_z, v, r);
        // big ? 0 : r0
        self.spv_select(u, big, zero, r0)
    }

    // ── int4 PackedU32 nibble access (8 signed nibbles per u32 word) ──────
    //
    // Element `idx` lives in word `idx/8`, nibble `idx%8`. Mirrors
    // `dtype::int4_{unpack,pack}`. Single-quark in the op-matrix; a packed
    // multi-quark store would need per-word ownership or atomics.

    /// Access-chain pointer to word `idx/8` of an int4 (u32-storage) field.
    fn i4_word_ptr(&mut self, var_id: u32, idx: u32) -> u32 {
        let u = self.spv_u32();
        let eight = self.spv_const(8);
        let word_idx = self.spv_bin(OP_UDIV, u, idx, eight);
        let zero = self.spv_const(0);
        let ptr_u32 = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, u);
        let chain = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ACCESS_CHAIN,
            &[ptr_u32, chain, var_id, zero, word_idx],
        );
        chain
    }

    /// `(idx % 8) * 4` — the bit shift of nibble `idx%8`.
    fn i4_nibble_shift(&mut self, idx: u32) -> u32 {
        let eight = self.spv_const(8);
        let four = self.spv_const(4);
        let u = self.spv_u32();
        let nib = self.spv_bin(OP_UMOD, u, idx, eight);
        self.spv_mul(nib, four)
    }

    /// Load + sign-extend the int4 at element `idx` → i32 value id.
    fn i4_load_nibble(&mut self, var_id: u32, idx: u32) -> u32 {
        let u = self.spv_u32();
        let chain = self.i4_word_ptr(var_id, idx);
        let word = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LOAD, &[u, word, chain, 0x2, 4]);
        let shift = self.i4_nibble_shift(idx);
        let shifted = self.spv_shr(word, shift);
        let mask = self.spv_const(0xF);
        let nib = self.spv_and(shifted, mask);
        // sign-extend 4-bit: (nib ^ 8) - 8
        let eight = self.spv_const(8);
        let xored = self.spv_bin(OP_BITWISE_XOR, u, nib, eight);
        let ext = self.spv_sub(xored, eight);
        // reinterpret the u32 bit pattern as i32 (same bits).
        let i32_ty = self.ensure_type_i32();
        let out = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[i32_ty, out, ext]);
        out
    }

    /// Read-modify-write the int4 nibble at element `idx` with i32 `val`.
    fn i4_store_nibble(&mut self, var_id: u32, idx: u32, val: u32) {
        let u = self.spv_u32();
        let chain = self.i4_word_ptr(var_id, idx);
        let word = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LOAD, &[u, word, chain, 0x2, 4]);
        let shift = self.i4_nibble_shift(idx);
        // val as u32 bits, masked to the low nibble.
        let val_u = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[u, val_u, val]);
        let mask4 = self.spv_const(0xF);
        let nib = self.spv_and(val_u, mask4);
        let nib_sh = self.spv_shl(nib, shift);
        // clear the target nibble: word & ~(0xF << shift). OpNot is unary.
        let lane_mask = self.spv_shl(mask4, shift);
        let not_mask = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_NOT, &[u, not_mask, lane_mask]);
        let cleared = self.spv_and(word, not_mask);
        let merged = self.spv_or(cleared, nib_sh);
        Self::emit_op(&mut self.sec_function, OP_STORE, &[chain, merged, 0x2, 4]);
    }
}
