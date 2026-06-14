//! KernelOp → SPIR-V instruction dispatch.
//!
//! The single `emit_single_op` method handles every KernelOp variant,
//! emitting the corresponding SPIR-V instruction(s) into sec_function.

use quanta_ir::*;

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
                        let ty = self.ensure_type_u32();
                        (self.emit_constant_u32(*v as u32), ty)
                    }
                    ConstValue::I32(v) => {
                        let ty = self.ensure_type_i32();
                        (self.emit_constant_i32(*v), ty)
                    }
                    ConstValue::I64(v) => {
                        let ty = self.ensure_type_i32();
                        (self.emit_constant_i32(*v as i32), ty)
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
                // Track integer constants for T1405 (Loop unroll on
                // small known counts). Match the truncation the SPIR-V
                // type system actually sees.
                match value {
                    ConstValue::U32(v) => {
                        self.reg_const_int.insert(dst.0, *v as i64);
                    }
                    ConstValue::U64(v) => {
                        self.reg_const_int.insert(dst.0, (*v as u32) as i64);
                    }
                    ConstValue::I32(v) => {
                        self.reg_const_int.insert(dst.0, *v as i64);
                    }
                    ConstValue::I64(v) => {
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
                self.emit_op_load(*dst, *field, *index, *ty)?;
            }

            KernelOp::Store {
                field,
                index,
                src,
                ty,
            } => {
                self.emit_op_store(*field, *index, *src, *ty)?;
            }

            KernelOp::BinOp { dst, a, b, op, ty } => {
                self.emit_op_binop(*dst, *a, *b, *op, *ty)?;
            }

            KernelOp::UnaryOp { dst, a, op, ty } => {
                self.emit_op_unary(*dst, *a, *op, *ty)?;
            }

            KernelOp::Cmp { dst, a, b, op, ty } => {
                self.emit_op_cmp(*dst, *a, *b, *op, *ty)?;
            }

            KernelOp::Cast { dst, src, from, to } => {
                self.emit_op_cast(*dst, *src, *from, *to)?;
            }

            KernelOp::Copy { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                self.set_reg(*dst, src_val, result_ty);
            }

            KernelOp::Branch {
                cond,
                then_ops,
                else_ops,
            } => {
                self.emit_op_branch(
                    *cond,
                    then_ops,
                    else_ops,
                    gid_var,
                    proton_id_var,
                    nucleus_id_var,
                    num_wg_var,
                )?;
            }

            KernelOp::Loop {
                count,
                iter_reg,
                body,
            } => {
                self.emit_op_loop(
                    *count,
                    *iter_reg,
                    body,
                    gid_var,
                    proton_id_var,
                    nucleus_id_var,
                    num_wg_var,
                )?;
            }

            KernelOp::Barrier => {
                let scope_wg = self.emit_constant_u32(SCOPE_WORKGROUP);
                let semantics =
                    self.emit_constant_u32(MEMORY_SEMANTICS_ACQ_REL | MEMORY_SEMANTICS_WORKGROUP);
                Self::emit_op(
                    &mut self.sec_function,
                    OP_CONTROL_BARRIER,
                    &[scope_wg, scope_wg, semantics],
                );
            }

            KernelOp::Fence { order } => {
                let order_bits: u32 = match order {
                    quanta_ir::MemoryOrder::Relaxed => 0,
                    quanta_ir::MemoryOrder::Acquire => MEMORY_SEMANTICS_ACQUIRE,
                    quanta_ir::MemoryOrder::Release => MEMORY_SEMANTICS_RELEASE,
                    quanta_ir::MemoryOrder::AcqRel => MEMORY_SEMANTICS_ACQ_REL,
                    quanta_ir::MemoryOrder::SeqCst => MEMORY_SEMANTICS_SEQ_CST,
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
                self.emit_op_shared_load(*dst, *id, *index, *ty)?;
            }

            KernelOp::SharedStore { id, index, src, .. } => {
                self.emit_op_shared_store(*id, *index, *src)?;
            }

            KernelOp::MathCall {
                dst,
                func,
                args,
                ty,
            } => {
                self.emit_op_math_call(*dst, *func, args, *ty)?;
            }

            KernelOp::Break => {
                if let Some(&merge_label) = self.loop_merge_stack.last() {
                    Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);
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
                self.emit_op_device_call(*dst, func_name, args, *ty)?;
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
                self.emit_op_atomic(*dst, *field, *index, *val, *op, *ty, *order)?;
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
                // SPIR-V `OpAtomicCompareExchange` takes a single scope/
                // semantics pair; we use `success_order` since LLVM's
                // constraint guarantees it dominates `failure_order`.
                self.emit_op_atomic_cas(
                    *dst,
                    *field,
                    *index,
                    *expected,
                    *desired,
                    *ty,
                    *success_order,
                )?;
            }

            KernelOp::SharedAtomicOp {
                dst,
                slot,
                index,
                val,
                op,
                ty,
                order,
            } => {
                self.emit_op_shared_atomic(*dst, *slot, *index, *val, *op, *ty, *order)?;
            }

            KernelOp::WaveShuffle {
                dst,
                src,
                lane_delta,
                ty,
            } => {
                self.emit_op_wave_shuffle(*dst, *src, *lane_delta, *ty)?;
            }

            KernelOp::WaveBallot { dst, predicate } => {
                self.emit_op_wave_ballot(*dst, *predicate)?;
            }

            KernelOp::WaveAny { dst, predicate } => {
                self.emit_op_wave_any(*dst, *predicate)?;
            }

            KernelOp::WaveAll { dst, predicate } => {
                self.emit_op_wave_all(*dst, *predicate)?;
            }

            KernelOp::TextureSample2D {
                dst,
                texture,
                x,
                y,
                ty,
            } => {
                self.emit_op_texture_sample_2d(*dst, *texture, *x, *y, *ty)?;
            }

            KernelOp::TextureSample3D { dst, ty, .. } => {
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_f32(0.0);
                self.set_reg(*dst, zero, result_ty);
            }

            KernelOp::TextureWrite2D {
                texture,
                x,
                y,
                value,
                ..
            } => {
                self.emit_op_texture_write_2d(*texture, *x, *y, *value)?;
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
                self.emit_op_subgroup_reduce(*dst, *src, *ty, false, false)?;
            }

            KernelOp::SubgroupReduceMin { dst, src, ty } => {
                self.emit_op_subgroup_minmax(*dst, *src, *ty, true)?;
            }

            KernelOp::SubgroupReduceMax { dst, src, ty } => {
                self.emit_op_subgroup_minmax(*dst, *src, *ty, false)?;
            }

            KernelOp::SubgroupExclusiveAdd { dst, src, ty } => {
                self.emit_op_subgroup_reduce(*dst, *src, *ty, true, false)?;
            }

            KernelOp::SubgroupInclusiveAdd { dst, src, ty } => {
                self.emit_op_subgroup_reduce(*dst, *src, *ty, false, true)?;
            }

            KernelOp::TextureLoad2D {
                dst,
                texture,
                x,
                y,
                ty,
            } => {
                self.emit_op_texture_load_2d(*dst, *texture, *x, *y, *ty)?;
            }

            KernelOp::SubgroupSize { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.emit_constant_u32(32);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::SharedDeclDyn { .. } => {
                // Handled during shared decl scan phase.
            }

            KernelOp::DebugPrint { src, ty } => {
                let _ = (src, ty);
            }

            KernelOp::Dispatch { .. } => {
                // Dynamic parallelism not supported in Vulkan compute
            }

            KernelOp::CooperativeMMA {
                dst, a, b, c, ty, ..
            } => {
                // Scalar fallback: D = A * B + C
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let c_val = self.reg_value_id(*c)?;
                let result_ty = self.scalar_type_id(*ty);
                let op_mul = if matches!(ty, ScalarType::F32 | ScalarType::F16) {
                    OP_FMUL
                } else {
                    OP_IMUL
                };
                let op_add = if matches!(ty, ScalarType::F32 | ScalarType::F16) {
                    OP_FADD
                } else {
                    OP_IADD
                };
                let mul = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    op_mul,
                    &[result_ty, mul, a_val, b_val],
                );
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    op_add,
                    &[result_ty, result, mul, c_val],
                );
                self.set_reg(*dst, result, result_ty);
            }
        }

        Ok(())
    }
}
