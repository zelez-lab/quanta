//! KernelOp dispatch: emit_ops / emit_op match.

use inkwell::types::BasicType;
use inkwell::values::BasicValueEnum;
use inkwell::{AddressSpace, AtomicOrdering, AtomicRMWBinOp, IntPredicate};

use quanta_ir::*;

use super::super::{
    EmitCtx, const_scalar_type, const_to_llvm, is_float_type, reg_load, reg_load_int, reg_store,
    scalar_to_llvm_type,
};
use super::helpers::{
    emit_binop, emit_cast, emit_cmp, emit_math_direct, emit_unary, make_vec_type,
};

pub(super) fn emit_ops<'a, 'ctx>(
    ectx: &mut EmitCtx<'a, 'ctx>,
    ops: &[KernelOp],
) -> Result<(), String> {
    for op in ops {
        emit_op(ectx, op)?;
    }
    Ok(())
}

fn emit_op<'a, 'ctx>(ectx: &mut EmitCtx<'a, 'ctx>, op: &KernelOp) -> Result<(), String> {
    match op {
        KernelOp::Const { dst, value } => {
            let val = const_to_llvm(ectx.context, value);
            let ty = const_scalar_type(value);
            reg_store(ectx.context, ectx.builder, ectx.reg_slots, dst.0, val, ty)?;
        }

        KernelOp::QuarkId { dst } => {
            let tid = ectx
                .intrinsics
                .thread_id_x(ectx.context, ectx.module, ectx.builder);
            let bid = ectx
                .intrinsics
                .block_id_x(ectx.context, ectx.module, ectx.builder);
            let bdim = ectx
                .intrinsics
                .block_dim_x(ectx.context, ectx.module, ectx.builder);
            let offset = ectx
                .builder
                .build_int_mul(bid, bdim, "")
                .map_err(|e| e.to_string())?;
            let gid = ectx
                .builder
                .build_int_add(offset, tid, "gid")
                .map_err(|e| e.to_string())?;
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                gid.into(),
                ScalarType::U32,
            )?;
        }

        KernelOp::ProtonId { dst } => {
            let tid = ectx
                .intrinsics
                .thread_id_x(ectx.context, ectx.module, ectx.builder);
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                tid.into(),
                ScalarType::U32,
            )?;
        }

        KernelOp::NucleusId { dst } => {
            let bid = ectx
                .intrinsics
                .block_id_x(ectx.context, ectx.module, ectx.builder);
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                bid.into(),
                ScalarType::U32,
            )?;
        }

        KernelOp::ProtonSize { dst } => {
            let bdim = ectx
                .intrinsics
                .block_dim_x(ectx.context, ectx.module, ectx.builder);
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                bdim.into(),
                ScalarType::U32,
            )?;
        }

        KernelOp::QuarkCount { dst } => {
            // Total threads = block_dim (approximate). Most kernels use quark_id() not quark_count().
            let bdim = ectx
                .intrinsics
                .block_dim_x(ectx.context, ectx.module, ectx.builder);
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                bdim.into(),
                ScalarType::U32,
            )?;
        }

        KernelOp::Load {
            dst,
            field,
            index,
            ty,
        } => {
            if index.0 == u32::MAX {
                if let Some((val, _)) = ectx.slot_to_const.get(field) {
                    reg_store(ectx.context, ectx.builder, ectx.reg_slots, dst.0, *val, *ty)?;
                }
            } else if let Some((ptr, scalar_ty)) = ectx.slot_to_arg.get(field) {
                let idx = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, index)?;
                let elem_ty = scalar_to_llvm_type(ectx.context, scalar_ty);
                let gep = unsafe {
                    ectx.builder
                        .build_gep(elem_ty, *ptr, &[idx], "")
                        .map_err(|e| e.to_string())?
                };
                let val = ectx
                    .builder
                    .build_load(elem_ty, gep, "load")
                    .map_err(|e| e.to_string())?;
                reg_store(
                    ectx.context,
                    ectx.builder,
                    ectx.reg_slots,
                    dst.0,
                    val,
                    *scalar_ty,
                )?;
            }
        }

        KernelOp::Store {
            field,
            index,
            src,
            ty: _,
        } => {
            if let Some((ptr, scalar_ty)) = ectx.slot_to_arg.get(field) {
                let idx = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, index)?;
                let val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, src.0)?;
                let elem_ty = scalar_to_llvm_type(ectx.context, scalar_ty);
                let gep = unsafe {
                    ectx.builder
                        .build_gep(elem_ty, *ptr, &[idx], "")
                        .map_err(|e| e.to_string())?
                };
                ectx.builder
                    .build_store(gep, val)
                    .map_err(|e| e.to_string())?;
            }
        }

        KernelOp::BinOp { dst, a, b, op, ty } => {
            let lhs = reg_load(ectx.context, ectx.builder, ectx.reg_slots, a.0)?;
            let rhs = reg_load(ectx.context, ectx.builder, ectx.reg_slots, b.0)?;
            let result = emit_binop(ectx.builder, lhs, rhs, op, ty)?;
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                result,
                *ty,
            )?;
        }

        KernelOp::Cmp { dst, a, b, op, ty } => {
            let lhs = reg_load(ectx.context, ectx.builder, ectx.reg_slots, a.0)?;
            let rhs = reg_load(ectx.context, ectx.builder, ectx.reg_slots, b.0)?;
            let result = emit_cmp(ectx.builder, lhs, rhs, op, ty)?;
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                result.into(),
                ScalarType::Bool,
            )?;
        }

        KernelOp::Cast { dst, src, from, to } => {
            let val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, src.0)?;
            let result = emit_cast(ectx.context, ectx.builder, val, from, to)?;
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                result,
                *to,
            )?;
        }

        KernelOp::MathCall {
            dst,
            func,
            args,
            ty,
        } => {
            let arg_vals: Vec<BasicValueEnum<'ctx>> = args
                .iter()
                .map(|r| reg_load(ectx.context, ectx.builder, ectx.reg_slots, r.0))
                .collect::<Result<Vec<_>, _>>()?;
            let result =
                emit_math_direct(ectx.context, ectx.module, ectx.builder, &arg_vals, func, ty)?;
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                result,
                *ty,
            )?;
        }

        KernelOp::Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            let cond_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, cond)?;
            let then_bb = ectx.context.append_basic_block(*ectx.function, "then");
            let else_bb = ectx.context.append_basic_block(*ectx.function, "else");
            let merge_bb = ectx.context.append_basic_block(*ectx.function, "merge");

            ectx.builder
                .build_conditional_branch(cond_val, then_bb, else_bb)
                .map_err(|e| e.to_string())?;

            ectx.builder.position_at_end(then_bb);
            emit_ops(ectx, then_ops)?;
            ectx.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| e.to_string())?;

            ectx.builder.position_at_end(else_bb);
            if !else_ops.is_empty() {
                emit_ops(ectx, else_ops)?;
            }
            ectx.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| e.to_string())?;

            ectx.builder.position_at_end(merge_bb);
        }

        KernelOp::Loop {
            count,
            iter_reg,
            body,
        } => {
            let header_bb = ectx
                .context
                .append_basic_block(*ectx.function, "loop_header");
            let body_bb = ectx.context.append_basic_block(*ectx.function, "loop_body");
            let exit_bb = ectx.context.append_basic_block(*ectx.function, "loop_exit");

            // Store initial iter = 0
            let i32_type = ectx.context.i32_type();
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                iter_reg.0,
                i32_type.const_zero().into(),
                ScalarType::U32,
            )?;

            ectx.builder
                .build_unconditional_branch(header_bb)
                .map_err(|e| e.to_string())?;

            // Header: load iter, compare with count
            ectx.builder.position_at_end(header_bb);
            let iter_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, iter_reg)?;
            let count_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, count)?;
            let cmp = ectx
                .builder
                .build_int_compare(IntPredicate::ULT, iter_val, count_val, "loop_cmp")
                .map_err(|e| e.to_string())?;
            ectx.builder
                .build_conditional_branch(cmp, body_bb, exit_bb)
                .map_err(|e| e.to_string())?;

            // Body
            ectx.builder.position_at_end(body_bb);
            emit_ops(ectx, body)?;

            // Increment iter
            let iter_val2 = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, iter_reg)?;
            let next = ectx
                .builder
                .build_int_add(iter_val2, i32_type.const_int(1, false), "next_iter")
                .map_err(|e| e.to_string())?;
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                iter_reg.0,
                next.into(),
                ScalarType::U32,
            )?;
            ectx.builder
                .build_unconditional_branch(header_bb)
                .map_err(|e| e.to_string())?;

            ectx.builder.position_at_end(exit_bb);
        }

        KernelOp::Barrier => {
            ectx.intrinsics
                .barrier(ectx.context, ectx.module, ectx.builder);
        }

        // LLVM `fence <ordering>` is a first-class IR instruction. The
        // mapping from `MemoryOrder` mirrors the AtomicOp arm below.
        // `Relaxed` has no LLVM-side fence (LLVM rejects `fence monotonic`),
        // so we emit a no-op for that arm — consistent with the
        // C11 / Rust `Ordering::Relaxed` semantics where a relaxed fence
        // is meaningless.
        KernelOp::Fence { order } => {
            let llvm_order = match order {
                quanta_ir::MemoryOrder::Relaxed => None,
                quanta_ir::MemoryOrder::Acquire => Some(AtomicOrdering::Acquire),
                quanta_ir::MemoryOrder::Release => Some(AtomicOrdering::Release),
                quanta_ir::MemoryOrder::AcqRel => Some(AtomicOrdering::AcquireRelease),
                quanta_ir::MemoryOrder::SeqCst => Some(AtomicOrdering::SequentiallyConsistent),
            };
            if let Some(ordering) = llvm_order {
                ectx.builder
                    .build_fence(ordering, /* is_single_thread */ false, "fence")
                    .map_err(|e| e.to_string())?;
            }
        }

        KernelOp::UnaryOp { dst, a, op, ty } => {
            let val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, a.0)?;
            let result = emit_unary(ectx.builder, val, op, ty)?;
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                result,
                *ty,
            )?;
        }

        KernelOp::SharedDecl { id, ty, count } => {
            // Create a global variable in address space 3 (shared/local memory)
            let elem_type = scalar_to_llvm_type(ectx.context, ty);
            let array_type = elem_type.array_type(*count);
            let global = ectx.module.add_global(
                array_type,
                Some(AddressSpace::from(3u16)),
                &format!("shared_{}", id),
            );
            global.set_initializer(&array_type.const_zero());
            ectx.shared_globals.insert(*id, global.as_pointer_value());
        }

        KernelOp::SharedLoad { dst, id, index, ty } => {
            let shared_ptr = ectx
                .shared_globals
                .get(id)
                .copied()
                .ok_or("shared memory not declared")?;
            let idx = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, index)?;
            let elem_type = scalar_to_llvm_type(ectx.context, ty);
            let gep = unsafe {
                ectx.builder
                    .build_gep(
                        elem_type,
                        shared_ptr,
                        &[ectx.context.i32_type().const_zero(), idx],
                        "shared_gep",
                    )
                    .map_err(|e| e.to_string())?
            };
            let val = ectx
                .builder
                .build_load(elem_type, gep, "shared_load")
                .map_err(|e| e.to_string())?;
            reg_store(ectx.context, ectx.builder, ectx.reg_slots, dst.0, val, *ty)?;
        }

        KernelOp::SharedStore { id, index, src, ty } => {
            let shared_ptr = ectx
                .shared_globals
                .get(id)
                .copied()
                .ok_or("shared memory not declared")?;
            let idx = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, index)?;
            let val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, src.0)?;
            let elem_type = scalar_to_llvm_type(ectx.context, ty);
            let gep = unsafe {
                ectx.builder
                    .build_gep(
                        elem_type,
                        shared_ptr,
                        &[ectx.context.i32_type().const_zero(), idx],
                        "shared_gep",
                    )
                    .map_err(|e| e.to_string())?
            };
            ectx.builder
                .build_store(gep, val)
                .map_err(|e| e.to_string())?;
        }

        KernelOp::Copy { dst, src, ty } => {
            let val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, src.0)?;
            reg_store(ectx.context, ectx.builder, ectx.reg_slots, dst.0, val, *ty)?;
        }

        // Quantization affine map — lowering lands in Phase B.
        KernelOp::Quantize { .. } | KernelOp::Dequantize { .. } => {
            return Err("LLVM: Quantize/Dequantize lowering pending".to_string());
        }

        KernelOp::Break => {
            // Break is handled at the Loop level — no-op here
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
            // Map the IR's MemoryOrder to inkwell's AtomicOrdering. LLVM's
            // `Monotonic` is the IR-level "Relaxed" — same semantics, just
            // the LLVM-spec name. We don't expose `Unordered` or
            // `NotAtomic` because the IR doesn't.
            let llvm_order = match order {
                quanta_ir::MemoryOrder::Relaxed => AtomicOrdering::Monotonic,
                quanta_ir::MemoryOrder::Acquire => AtomicOrdering::Acquire,
                quanta_ir::MemoryOrder::Release => AtomicOrdering::Release,
                quanta_ir::MemoryOrder::AcqRel => AtomicOrdering::AcquireRelease,
                quanta_ir::MemoryOrder::SeqCst => AtomicOrdering::SequentiallyConsistent,
            };
            if let Some((ptr, scalar_ty)) = ectx.slot_to_arg.get(field) {
                let idx = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, index)?;
                let elem_ty = scalar_to_llvm_type(ectx.context, scalar_ty);
                let gep = unsafe {
                    ectx.builder
                        .build_gep(elem_ty, *ptr, &[idx], "atomic_ptr")
                        .map_err(|e| e.to_string())?
                };

                let is_float = is_float_type(ty);

                if *op == quanta_ir::AtomicOp::CompareExchange {
                    // CompareExchange via atomicrmw is not standard; use cmpxchg.
                    // For AtomicOp::CompareExchange, `val` holds expected, but we
                    // don't have a separate desired. Treat as exchange (xchg) instead.
                    // The proper CAS path is KernelOp::AtomicCas.
                    let value = reg_load(ectx.context, ectx.builder, ectx.reg_slots, val.0)?;
                    if is_float {
                        let int_ty = match ty {
                            ScalarType::F32 => ectx.context.i32_type(),
                            ScalarType::F64 => ectx.context.i64_type(),
                            _ => ectx.context.i16_type(), // F16
                        };
                        let val_as_int = ectx
                            .builder
                            .build_bit_cast(value, int_ty, "atomic_f2i")
                            .map_err(|e| e.to_string())?
                            .into_int_value();
                        let result = ectx
                            .builder
                            .build_atomicrmw(AtomicRMWBinOp::Xchg, gep, val_as_int, llvm_order)
                            .map_err(|e| e.to_string())?;
                        let result_float = ectx
                            .builder
                            .build_bit_cast(result, elem_ty, "atomic_i2f")
                            .map_err(|e| e.to_string())?;
                        reg_store(
                            ectx.context,
                            ectx.builder,
                            ectx.reg_slots,
                            dst.0,
                            result_float,
                            *ty,
                        )?;
                    } else {
                        let val_int = value.into_int_value();
                        let result = ectx
                            .builder
                            .build_atomicrmw(AtomicRMWBinOp::Xchg, gep, val_int, llvm_order)
                            .map_err(|e| e.to_string())?;
                        reg_store(
                            ectx.context,
                            ectx.builder,
                            ectx.reg_slots,
                            dst.0,
                            result.into(),
                            *ty,
                        )?;
                    }
                } else if is_float {
                    // Float atomics: use FAdd/FSub for add/sub, bitcast for others
                    let value = reg_load(ectx.context, ectx.builder, ectx.reg_slots, val.0)?;
                    let int_ty = match ty {
                        ScalarType::F32 => ectx.context.i32_type(),
                        ScalarType::F64 => ectx.context.i64_type(),
                        _ => ectx.context.i16_type(), // F16
                    };

                    match op {
                        quanta_ir::AtomicOp::Add | quanta_ir::AtomicOp::Sub => {
                            // inkwell's build_atomicrmw only takes IntValue, so for
                            // float add/sub we bitcast to int, issue the op, bitcast back.
                            // LLVM itself supports atomicrmw fadd/fsub on float types,
                            // but inkwell's Rust API restricts to IntValue.
                            let val_as_int = ectx
                                .builder
                                .build_bit_cast(value, int_ty, "atomic_f2i")
                                .map_err(|e| e.to_string())?
                                .into_int_value();
                            let rmw_op = if *op == quanta_ir::AtomicOp::Add {
                                AtomicRMWBinOp::FAdd
                            } else {
                                AtomicRMWBinOp::FSub
                            };
                            let result = ectx
                                .builder
                                .build_atomicrmw(rmw_op, gep, val_as_int, llvm_order)
                                .map_err(|e| e.to_string())?;
                            let result_float = ectx
                                .builder
                                .build_bit_cast(result, elem_ty, "atomic_i2f")
                                .map_err(|e| e.to_string())?;
                            reg_store(
                                ectx.context,
                                ectx.builder,
                                ectx.reg_slots,
                                dst.0,
                                result_float,
                                *ty,
                            )?;
                        }
                        quanta_ir::AtomicOp::Exchange => {
                            let val_as_int = ectx
                                .builder
                                .build_bit_cast(value, int_ty, "atomic_f2i")
                                .map_err(|e| e.to_string())?
                                .into_int_value();
                            let result = ectx
                                .builder
                                .build_atomicrmw(AtomicRMWBinOp::Xchg, gep, val_as_int, llvm_order)
                                .map_err(|e| e.to_string())?;
                            let result_float = ectx
                                .builder
                                .build_bit_cast(result, elem_ty, "atomic_i2f")
                                .map_err(|e| e.to_string())?;
                            reg_store(
                                ectx.context,
                                ectx.builder,
                                ectx.reg_slots,
                                dst.0,
                                result_float,
                                *ty,
                            )?;
                        }
                        _ => {
                            return Err(format!("AtomicOp {:?} not supported on float types", op));
                        }
                    }
                } else {
                    // Integer atomics
                    let value = reg_load(ectx.context, ectx.builder, ectx.reg_slots, val.0)?;
                    let val_int = value.into_int_value();
                    let is_signed = matches!(
                        ty,
                        ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                    );
                    let rmw_op = match op {
                        quanta_ir::AtomicOp::Add => AtomicRMWBinOp::Add,
                        quanta_ir::AtomicOp::Sub => AtomicRMWBinOp::Sub,
                        quanta_ir::AtomicOp::Min => {
                            if is_signed {
                                AtomicRMWBinOp::Min
                            } else {
                                AtomicRMWBinOp::UMin
                            }
                        }
                        quanta_ir::AtomicOp::Max => {
                            if is_signed {
                                AtomicRMWBinOp::Max
                            } else {
                                AtomicRMWBinOp::UMax
                            }
                        }
                        quanta_ir::AtomicOp::And => AtomicRMWBinOp::And,
                        quanta_ir::AtomicOp::Or => AtomicRMWBinOp::Or,
                        quanta_ir::AtomicOp::Xor => AtomicRMWBinOp::Xor,
                        quanta_ir::AtomicOp::Exchange => AtomicRMWBinOp::Xchg,
                        quanta_ir::AtomicOp::CompareExchange => unreachable!(),
                    };
                    let result = ectx
                        .builder
                        .build_atomicrmw(rmw_op, gep, val_int, llvm_order)
                        .map_err(|e| e.to_string())?;
                    reg_store(
                        ectx.context,
                        ectx.builder,
                        ectx.reg_slots,
                        dst.0,
                        result.into(),
                        *ty,
                    )?;
                }
            }
        }

        KernelOp::AtomicCas {
            dst,
            field,
            index,
            expected,
            desired,
            ty,
            success_order,
            failure_order,
        } => {
            // LLVM `cmpxchg` takes two orderings (success / failure)
            // with the constraints `failure ≤ success` and
            // `failure ∉ {Release, AcqRel}`. Map both IR fields to
            // LLVM AtomicOrdering, then clamp `failure` if a caller
            // chose an invalid combination.
            let map_order = |order: &quanta_ir::MemoryOrder| -> AtomicOrdering {
                match order {
                    quanta_ir::MemoryOrder::Relaxed => AtomicOrdering::Monotonic,
                    quanta_ir::MemoryOrder::Acquire => AtomicOrdering::Acquire,
                    quanta_ir::MemoryOrder::Release => AtomicOrdering::Release,
                    quanta_ir::MemoryOrder::AcqRel => AtomicOrdering::AcquireRelease,
                    quanta_ir::MemoryOrder::SeqCst => AtomicOrdering::SequentiallyConsistent,
                }
            };
            let llvm_success = map_order(success_order);
            // Clamp failure: never Release/AcqRel; if SeqCst on success,
            // failure must be SeqCst or weaker.
            let llvm_failure = match failure_order {
                quanta_ir::MemoryOrder::Release | quanta_ir::MemoryOrder::AcqRel => {
                    AtomicOrdering::Acquire
                }
                _ => map_order(failure_order),
            };
            if let Some((ptr, scalar_ty)) = ectx.slot_to_arg.get(field) {
                let idx = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, index)?;
                let elem_ty = scalar_to_llvm_type(ectx.context, scalar_ty);
                let gep = unsafe {
                    ectx.builder
                        .build_gep(elem_ty, *ptr, &[idx], "cas_ptr")
                        .map_err(|e| e.to_string())?
                };

                let exp_val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, expected.0)?;
                let des_val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, desired.0)?;

                if is_float_type(ty) {
                    // cmpxchg requires integer or pointer operands; bitcast floats
                    let int_ty = match ty {
                        ScalarType::F32 => ectx.context.i32_type(),
                        ScalarType::F64 => ectx.context.i64_type(),
                        _ => ectx.context.i16_type(), // F16
                    };
                    let exp_int = ectx
                        .builder
                        .build_bit_cast(exp_val, int_ty, "cas_exp_f2i")
                        .map_err(|e| e.to_string())?
                        .into_int_value();
                    let des_int = ectx
                        .builder
                        .build_bit_cast(des_val, int_ty, "cas_des_f2i")
                        .map_err(|e| e.to_string())?
                        .into_int_value();
                    let result = ectx
                        .builder
                        .build_cmpxchg(gep, exp_int, des_int, llvm_success, llvm_failure)
                        .map_err(|e| e.to_string())?;
                    let old_int = ectx
                        .builder
                        .build_extract_value(result, 0, "cas_old")
                        .map_err(|e| e.to_string())?;
                    let old_float = ectx
                        .builder
                        .build_bit_cast(old_int, elem_ty, "cas_i2f")
                        .map_err(|e| e.to_string())?;
                    reg_store(
                        ectx.context,
                        ectx.builder,
                        ectx.reg_slots,
                        dst.0,
                        old_float,
                        *ty,
                    )?;
                } else {
                    let exp_int = exp_val.into_int_value();
                    let des_int = des_val.into_int_value();
                    let result = ectx
                        .builder
                        .build_cmpxchg(gep, exp_int, des_int, llvm_success, llvm_failure)
                        .map_err(|e| e.to_string())?;
                    let old_val = ectx
                        .builder
                        .build_extract_value(result, 0, "cas_old")
                        .map_err(|e| e.to_string())?;
                    reg_store(
                        ectx.context,
                        ectx.builder,
                        ectx.reg_slots,
                        dst.0,
                        old_val,
                        *ty,
                    )?;
                }
            }
        }
        // CPU-JIT (LLVM) lane: shared memory is currently emulated as
        // a per-thread scratch buffer with no cross-lane atomicity
        // guarantees (single-thread JIT runs are not multi-lane).
        // Shared-memory atomics therefore have no meaningful LLVM
        // lowering today; refuse with a clear error rather than
        // silently emit a non-atomic store-load pair.
        KernelOp::SharedAtomicOp { .. } => {
            return Err(
                "shared-memory atomics not yet supported by the CPU-JIT (LLVM) backend; \
                 use a buffer-backed atomic counter as a fallback"
                    .to_string(),
            );
        }
        KernelOp::WaveShuffle {
            dst,
            src,
            lane_delta,
            ty,
        } => {
            let src_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, src)?;
            let delta_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, lane_delta)?;
            let result = ectx.intrinsics.wave_shuffle(
                ectx.context,
                ectx.module,
                ectx.builder,
                src_val,
                delta_val,
            );
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                result.into(),
                *ty,
            )?;
        }
        KernelOp::WaveBallot { dst, predicate } => {
            let pred_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, predicate)?;
            let result =
                ectx.intrinsics
                    .wave_ballot(ectx.context, ectx.module, ectx.builder, pred_val);
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                result.into(),
                ScalarType::U32,
            )?;
        }
        KernelOp::WaveAny { dst, predicate } => {
            let pred_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, predicate)?;
            let result =
                ectx.intrinsics
                    .wave_any(ectx.context, ectx.module, ectx.builder, pred_val);
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                result.into(),
                ScalarType::U32,
            )?;
        }
        KernelOp::WaveAll { dst, predicate } => {
            let pred_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, predicate)?;
            let result =
                ectx.intrinsics
                    .wave_all(ectx.context, ectx.module, ectx.builder, pred_val);
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst.0,
                result.into(),
                ScalarType::U32,
            )?;
        }
        KernelOp::VecConstruct {
            dst,
            components,
            ty,
        } => {
            let n = components.len() as u32;
            let scalar_llvm = scalar_to_llvm_type(ectx.context, ty);
            let vec_ty = make_vec_type(scalar_llvm, n);

            // Start with undef and insert each component
            let mut vec_val = vec_ty.get_undef();
            for (i, comp) in components.iter().enumerate() {
                let comp_val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, comp.0)?;
                let idx = ectx.context.i32_type().const_int(i as u64, false);
                vec_val = ectx
                    .builder
                    .build_insert_element(vec_val, comp_val, idx, "")
                    .map_err(|e| e.to_string())?;
            }

            // Create a vector-typed alloca and store the constructed vector
            let alloca = ectx
                .builder
                .build_alloca(vec_ty, &format!("r{}", dst.0))
                .map_err(|e| e.to_string())?;
            ectx.builder
                .build_store(alloca, vec_val)
                .map_err(|e| e.to_string())?;
            ectx.reg_slots.insert(dst.0, (alloca, *ty));
        }
        KernelOp::VecExtract {
            dst,
            vec,
            component,
            ty,
        } => {
            // Load the vector value from the source register's alloca.
            // The alloca was created by VecConstruct and holds a vector type.
            let (vec_ptr, scalar_ty) = ectx
                .reg_slots
                .get(&vec.0)
                .ok_or_else(|| format!("register r{} not allocated (VecExtract source)", vec.0))?;
            let scalar_llvm = scalar_to_llvm_type(ectx.context, scalar_ty);
            // Determine vector width from the component index.  VecConstruct
            // created the alloca with the exact width, but we don't store that
            // width in reg_slots.  Use component+1 as a lower bound, clamped
            // to the common GPU vector sizes (2, 3, 4).
            let vec_width = if *component < 2 {
                2
            } else {
                (*component as u32) + 1
            };
            let vec_ty = make_vec_type(scalar_llvm, vec_width);
            let vec_val = ectx
                .builder
                .build_load(vec_ty, *vec_ptr, "vec_load")
                .map_err(|e| e.to_string())?;
            let idx = ectx.context.i32_type().const_int(*component as u64, false);
            let elem = ectx
                .builder
                .build_extract_element(vec_val.into_vector_value(), idx, "vec_extract")
                .map_err(|e| e.to_string())?;
            reg_store(ectx.context, ectx.builder, ectx.reg_slots, dst.0, elem, *ty)?;
        }
        KernelOp::MatMul {
            dst,
            a,
            b,
            size,
            ty,
        } => {
            let n = *size as u32;
            let n2 = n * n;

            // Load source matrices (flat vectors of n*n elements)
            let (a_ptr, a_scalar) = ectx
                .reg_slots
                .get(&a.0)
                .ok_or_else(|| format!("register r{} not allocated (MatMul a)", a.0))?;
            let a_scalar_llvm = scalar_to_llvm_type(ectx.context, a_scalar);
            let a_vec_ty = make_vec_type(a_scalar_llvm, n2);
            let a_vec = ectx
                .builder
                .build_load(a_vec_ty, *a_ptr, "matmul_a")
                .map_err(|e| e.to_string())?
                .into_vector_value();

            let (b_ptr, b_scalar) = ectx
                .reg_slots
                .get(&b.0)
                .ok_or_else(|| format!("register r{} not allocated (MatMul b)", b.0))?;
            let b_scalar_llvm = scalar_to_llvm_type(ectx.context, b_scalar);
            let b_vec_ty = make_vec_type(b_scalar_llvm, n2);
            let b_vec = ectx
                .builder
                .build_load(b_vec_ty, *b_ptr, "matmul_b")
                .map_err(|e| e.to_string())?
                .into_vector_value();

            // Build result vector: result[i*n+j] = sum_k(a[i*n+k] * b[k*n+j])
            let scalar_llvm = scalar_to_llvm_type(ectx.context, ty);
            let result_vec_ty = make_vec_type(scalar_llvm, n2);
            let is_float = is_float_type(ty);
            let i32_ty = ectx.context.i32_type();

            let mut result_vec = result_vec_ty.get_undef();

            for i in 0..n {
                for j in 0..n {
                    // Accumulate: sum = a[i*n+0]*b[0*n+j] + a[i*n+1]*b[1*n+j] + ...
                    let mut acc: Option<BasicValueEnum<'ctx>> = None;
                    for k in 0..n {
                        let a_idx = i32_ty.const_int((i * n + k) as u64, false);
                        let b_idx = i32_ty.const_int((k * n + j) as u64, false);

                        let a_elem = ectx
                            .builder
                            .build_extract_element(a_vec, a_idx, "a_elem")
                            .map_err(|e| e.to_string())?;
                        let b_elem = ectx
                            .builder
                            .build_extract_element(b_vec, b_idx, "b_elem")
                            .map_err(|e| e.to_string())?;

                        let prod = if is_float {
                            ectx.builder
                                .build_float_mul(
                                    a_elem.into_float_value(),
                                    b_elem.into_float_value(),
                                    "mul",
                                )
                                .map_err(|e| e.to_string())?
                                .into()
                        } else {
                            ectx.builder
                                .build_int_mul(
                                    a_elem.into_int_value(),
                                    b_elem.into_int_value(),
                                    "mul",
                                )
                                .map_err(|e| e.to_string())?
                                .into()
                        };

                        acc = Some(match acc {
                            None => prod,
                            Some(prev) => {
                                if is_float {
                                    ectx.builder
                                        .build_float_add(
                                            prev.into_float_value(),
                                            prod.into_float_value(),
                                            "acc",
                                        )
                                        .map_err(|e| e.to_string())?
                                        .into()
                                } else {
                                    ectx.builder
                                        .build_int_add(
                                            prev.into_int_value(),
                                            prod.into_int_value(),
                                            "acc",
                                        )
                                        .map_err(|e| e.to_string())?
                                        .into()
                                }
                            }
                        });
                    }

                    let r_idx = i32_ty.const_int((i * n + j) as u64, false);
                    result_vec = ectx
                        .builder
                        .build_insert_element(result_vec, acc.unwrap(), r_idx, "")
                        .map_err(|e| e.to_string())?;
                }
            }

            // Store result as a vector-typed alloca
            let alloca = ectx
                .builder
                .build_alloca(result_vec_ty, &format!("r{}", dst.0))
                .map_err(|e| e.to_string())?;
            ectx.builder
                .build_store(alloca, result_vec)
                .map_err(|e| e.to_string())?;
            ectx.reg_slots.insert(dst.0, (alloca, *ty));
        }

        KernelOp::TextureSample2D {
            dst,
            texture,
            x,
            y,
            ty,
        } => {
            // Get texture handle (i32) from slot_to_const
            let tex_handle = ectx
                .slot_to_const
                .get(texture)
                .ok_or_else(|| format!("texture slot {} not bound", texture))?
                .0
                .into_int_value();

            let x_val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, x.0)?;
            let y_val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, y.0)?;

            let result = ectx.intrinsics.texture_sample_2d(
                ectx.context,
                ectx.module,
                ectx.builder,
                tex_handle,
                x_val,
                y_val,
            );

            // Result is vec4. Store as a vector-typed alloca so VecExtract can read it.
            let f32_type = ectx.context.f32_type();
            let vec4_ty = f32_type.vec_type(4);
            let alloca = ectx
                .builder
                .build_alloca(vec4_ty, &format!("r{}", dst.0))
                .map_err(|e| e.to_string())?;
            ectx.builder
                .build_store(alloca, result)
                .map_err(|e| e.to_string())?;
            ectx.reg_slots.insert(dst.0, (alloca, *ty));
        }

        KernelOp::TextureSample3D {
            dst,
            texture,
            x,
            y,
            z,
            ty,
        } => {
            let tex_handle = ectx
                .slot_to_const
                .get(texture)
                .ok_or_else(|| format!("texture slot {} not bound", texture))?
                .0
                .into_int_value();

            let x_val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, x.0)?;
            let y_val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, y.0)?;
            let z_val = reg_load(ectx.context, ectx.builder, ectx.reg_slots, z.0)?;

            let result = ectx.intrinsics.texture_sample_3d(
                ectx.context,
                ectx.module,
                ectx.builder,
                tex_handle,
                x_val,
                y_val,
                z_val,
            );

            let f32_type = ectx.context.f32_type();
            let vec4_ty = f32_type.vec_type(4);
            let alloca = ectx
                .builder
                .build_alloca(vec4_ty, &format!("r{}", dst.0))
                .map_err(|e| e.to_string())?;
            ectx.builder
                .build_store(alloca, result)
                .map_err(|e| e.to_string())?;
            ectx.reg_slots.insert(dst.0, (alloca, *ty));
        }

        KernelOp::TextureWrite2D {
            texture,
            x,
            y,
            value,
            ty: _,
        } => {
            let tex_handle = ectx
                .slot_to_const
                .get(texture)
                .ok_or_else(|| format!("texture slot {} not bound", texture))?
                .0
                .into_int_value();

            let x_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, x)?;
            let y_val = reg_load_int(ectx.context, ectx.builder, ectx.reg_slots, y)?;

            // Load the vec4 value from the source register
            let (val_ptr, val_scalar) = ectx.reg_slots.get(&value.0).ok_or_else(|| {
                format!("register r{} not allocated (TextureWrite2D value)", value.0)
            })?;
            let val_scalar_llvm = scalar_to_llvm_type(ectx.context, val_scalar);
            let vec4_ty = make_vec_type(val_scalar_llvm, 4);
            let vec_val = ectx
                .builder
                .build_load(vec4_ty, *val_ptr, "tex_write_val")
                .map_err(|e| e.to_string())?;

            ectx.intrinsics.texture_write_2d(
                ectx.context,
                ectx.module,
                ectx.builder,
                tex_handle,
                x_val,
                y_val,
                vec_val,
            );
        }

        KernelOp::TextureSize {
            dst_w,
            dst_h,
            texture,
        } => {
            let tex_handle = ectx
                .slot_to_const
                .get(texture)
                .ok_or_else(|| format!("texture slot {} not bound", texture))?
                .0
                .into_int_value();

            let (width, height) = ectx.intrinsics.texture_size_2d(
                ectx.context,
                ectx.module,
                ectx.builder,
                tex_handle,
            );

            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst_w.0,
                width.into(),
                ScalarType::U32,
            )?;
            reg_store(
                ectx.context,
                ectx.builder,
                ectx.reg_slots,
                dst_h.0,
                height.into(),
                ScalarType::U32,
            )?;
        }
        KernelOp::Dispatch { .. } => {
            return Err("dynamic parallelism (Dispatch) not supported".into());
        }
        KernelOp::DeviceCall { .. } => {
            // DeviceCall is handled by the text-based MSL/WGSL emitters.
            // In the LLVM path, device functions are compiled from their Rust source
            // via the rustc path and inlined by LLVM. The KernelOp IR path should not
            // see DeviceCall ops — they are resolved at the source level.
            return Err("DeviceCall not supported in LLVM KernelOp path (use rustc path)".into());
        }
        KernelOp::Bitcast { .. }
        | KernelOp::CountTrailingZeros { .. }
        | KernelOp::CountLeadingZeros { .. }
        | KernelOp::PopCount { .. }
        | KernelOp::Dot { .. }
        | KernelOp::SubgroupReduceAdd { .. }
        | KernelOp::SubgroupReduceMin { .. }
        | KernelOp::SubgroupReduceMax { .. }
        | KernelOp::SubgroupExclusiveAdd { .. }
        | KernelOp::SubgroupInclusiveAdd { .. }
        | KernelOp::TextureLoad2D { .. }
        | KernelOp::SubgroupSize { .. }
        | KernelOp::SharedDeclDyn { .. }
        | KernelOp::DebugPrint { .. }
        | KernelOp::CooperativeMMA { .. }
        | KernelOp::CooperativeMatrixLoad { .. }
        | KernelOp::CooperativeMatrixStore { .. } => {
            return Err(
                "new IR ops (bitcast, CTZ/CLZ, popcount, dot, subgroup, texture load, shared dyn, debug print, cooperative matrix) not yet supported in LLVM path"
                    .into(),
            );
        }
    }
    Ok(())
}
