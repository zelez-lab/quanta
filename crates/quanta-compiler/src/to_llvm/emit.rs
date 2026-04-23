//! Kernel building and op emission.

use std::collections::HashMap;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicType, BasicTypeEnum, VectorType};
use inkwell::values::{BasicValueEnum, PointerValue};
use inkwell::{AddressSpace, AtomicOrdering, AtomicRMWBinOp, FloatPredicate, IntPredicate};

use crate::targets::{GpuIntrinsics, GpuTarget};
use quanta_ir::*;

use super::metadata::{add_nvptx_kernel_metadata, add_spirv_compute_metadata};
use super::{
    EmitCtx, const_scalar_type, const_to_llvm, is_float_type, reg_load, reg_load_int, reg_store,
    scalar_to_llvm_type,
};

pub(crate) fn build_kernel<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    kernel: &KernelDef,
    intrinsics: &dyn GpuIntrinsics<'ctx>,
    target: GpuTarget,
) -> Result<(), String> {
    // Address space 1 = global memory (GPU)
    let global_as = AddressSpace::from(1u16);

    // Build parameter types
    let mut param_types: Vec<inkwell::types::BasicTypeEnum<'ctx>> = Vec::new();
    for param in &kernel.params {
        match param {
            KernelParam::FieldRead { scalar_type: _, .. }
            | KernelParam::FieldWrite { scalar_type: _, .. } => {
                // Pointer to global memory
                param_types.push(context.ptr_type(global_as).into());
            }
            KernelParam::Constant { scalar_type, .. } => {
                param_types.push(scalar_to_llvm_type(context, scalar_type));
            }
            KernelParam::Texture2DRead { scalar_type: _, .. }
            | KernelParam::Texture2DWrite { scalar_type: _, .. }
            | KernelParam::Texture3DRead { scalar_type: _, .. } => {
                // Texture handles are passed as i32 descriptor indices
                param_types.push(context.i32_type().into());
            }
        }
    }

    // Create function
    let fn_type = context.void_type().fn_type(
        &param_types.iter().map(|t| (*t).into()).collect::<Vec<_>>(),
        false,
    );
    let function = module.add_function(&kernel.name, fn_type, None);

    // Set calling convention for AMD kernels
    let cc = intrinsics.kernel_calling_convention();
    if cc != 0 {
        function.set_call_conventions(cc);
    }

    // Entry block
    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    // Register file -- alloca-based (LLVM mem2reg promotes to SSA with phi nodes)
    // This avoids SSA dominance issues when registers are written inside loops/branches.
    let mut reg_slots: HashMap<u32, (PointerValue<'ctx>, ScalarType)> = HashMap::new();

    // Pre-allocate register slots for all registers used in the kernel
    for reg_id in 0..kernel.next_reg {
        let ty = context.f32_type(); // default -- will be overwritten on first store
        let alloca = builder
            .build_alloca(ty, &format!("r{}", reg_id))
            .map_err(|e| e.to_string())?;
        reg_slots.insert(reg_id, (alloca, ScalarType::F32));
    }

    // Map param slots to function arguments
    let mut slot_to_arg: HashMap<u32, (PointerValue<'ctx>, ScalarType)> = HashMap::new();
    let mut slot_to_const: HashMap<u32, (BasicValueEnum<'ctx>, ScalarType)> = HashMap::new();
    let mut arg_idx = 0u32;
    for param in &kernel.params {
        match param {
            KernelParam::FieldRead {
                slot, scalar_type, ..
            }
            | KernelParam::FieldWrite {
                slot, scalar_type, ..
            } => {
                let ptr = function
                    .get_nth_param(arg_idx)
                    .unwrap()
                    .into_pointer_value();
                slot_to_arg.insert(*slot, (ptr, *scalar_type));
                arg_idx += 1;
            }
            KernelParam::Constant {
                slot, scalar_type, ..
            } => {
                let val = function.get_nth_param(arg_idx).unwrap();
                slot_to_const.insert(*slot, (val, *scalar_type));
                arg_idx += 1;
            }
            KernelParam::Texture2DRead {
                slot, scalar_type, ..
            }
            | KernelParam::Texture2DWrite {
                slot, scalar_type, ..
            }
            | KernelParam::Texture3DRead {
                slot, scalar_type, ..
            } => {
                // Texture handles are i32 values, store as scalar constants
                let val = function.get_nth_param(arg_idx).unwrap();
                slot_to_const.insert(*slot, (val, *scalar_type));
                arg_idx += 1;
            }
        }
    }

    // Shared memory globals (populated by SharedDecl ops)
    let mut shared_globals: HashMap<u32, PointerValue<'ctx>> = HashMap::new();

    // Emit ops
    let mut ectx = EmitCtx {
        context,
        module,
        builder,
        function: &function,
        reg_slots: &mut reg_slots,
        slot_to_arg: &slot_to_arg,
        slot_to_const: &slot_to_const,
        intrinsics,
        _target: target,
        shared_globals: &mut shared_globals,
    };
    emit_ops(&mut ectx, &kernel.body)?;

    // Return void
    builder.build_return(None).map_err(|e| e.to_string())?;

    // Add target-specific kernel metadata
    if target == GpuTarget::Nvptx {
        add_nvptx_kernel_metadata(context, module, &function);
    } else if target == GpuTarget::Spirv {
        add_spirv_compute_metadata(context, &function);
    }

    Ok(())
}

fn emit_ops<'a, 'ctx>(ectx: &mut EmitCtx<'a, 'ctx>, ops: &[KernelOp]) -> Result<(), String> {
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

        KernelOp::LocalId { dst } => {
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

        KernelOp::GroupId { dst } => {
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

        KernelOp::GroupSize { dst } => {
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
        } => {
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
                            .build_atomicrmw(
                                AtomicRMWBinOp::Xchg,
                                gep,
                                val_as_int,
                                AtomicOrdering::Monotonic,
                            )
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
                            .build_atomicrmw(
                                AtomicRMWBinOp::Xchg,
                                gep,
                                val_int,
                                AtomicOrdering::Monotonic,
                            )
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
                                .build_atomicrmw(rmw_op, gep, val_as_int, AtomicOrdering::Monotonic)
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
                                .build_atomicrmw(
                                    AtomicRMWBinOp::Xchg,
                                    gep,
                                    val_as_int,
                                    AtomicOrdering::Monotonic,
                                )
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
                        .build_atomicrmw(rmw_op, gep, val_int, AtomicOrdering::Monotonic)
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
        } => {
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
                        .build_cmpxchg(
                            gep,
                            exp_int,
                            des_int,
                            AtomicOrdering::Monotonic,
                            AtomicOrdering::Monotonic,
                        )
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
                        .build_cmpxchg(
                            gep,
                            exp_int,
                            des_int,
                            AtomicOrdering::Monotonic,
                            AtomicOrdering::Monotonic,
                        )
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
        | KernelOp::CooperativeMMA { .. } => {
            return Err(
                "new IR ops (bitcast, CTZ/CLZ, popcount, dot, subgroup, texture load, shared dyn, debug print) not yet supported in LLVM path"
                    .into(),
            );
        }
    }
    Ok(())
}

// ============================================================================
// Op-level helpers
// ============================================================================

fn emit_binop<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    op: &BinOp,
    ty: &ScalarType,
) -> Result<BasicValueEnum<'ctx>, String> {
    if is_float_type(ty) {
        let a = lhs.into_float_value();
        let b = rhs.into_float_value();
        let r = match op {
            BinOp::Add => builder.build_float_add(a, b, ""),
            BinOp::Sub => builder.build_float_sub(a, b, ""),
            BinOp::Mul => builder.build_float_mul(a, b, ""),
            BinOp::Div => builder.build_float_div(a, b, ""),
            BinOp::Rem => builder.build_float_rem(a, b, ""),
            BinOp::SatAdd => builder.build_float_add(a, b, ""), // float doesn't overflow
            BinOp::SatSub => builder.build_float_sub(a, b, ""), // float doesn't overflow
            _ => return Err("bitwise ops not supported on floats".into()),
        }
        .map_err(|e| e.to_string())?;
        Ok(r.into())
    } else {
        let a = lhs.into_int_value();
        let b = rhs.into_int_value();
        let r = match op {
            BinOp::Add => builder.build_int_add(a, b, ""),
            BinOp::Sub => builder.build_int_sub(a, b, ""),
            BinOp::Mul => builder.build_int_mul(a, b, ""),
            BinOp::Div => builder.build_int_unsigned_div(a, b, ""),
            BinOp::Rem => builder.build_int_unsigned_rem(a, b, ""),
            BinOp::BitAnd => builder.build_and(a, b, ""),
            BinOp::BitOr => builder.build_or(a, b, ""),
            BinOp::BitXor => builder.build_xor(a, b, ""),
            BinOp::Shl => builder.build_left_shift(a, b, ""),
            BinOp::Shr => builder.build_right_shift(a, b, false, ""),
            BinOp::SatAdd => {
                // Saturating add: add then clamp overflow
                let sum = builder.build_int_add(a, b, "").map_err(|e| e.to_string())?;
                // Unsigned overflow: sum < a
                let overflow = builder
                    .build_int_compare(inkwell::IntPredicate::ULT, sum, a, "")
                    .map_err(|e| e.to_string())?;
                let max_val = a.get_type().const_all_ones();
                return builder
                    .build_select(overflow, max_val, sum, "")
                    .map_err(|e| e.to_string());
            }
            BinOp::SatSub => {
                let diff = builder.build_int_sub(a, b, "").map_err(|e| e.to_string())?;
                let underflow = builder
                    .build_int_compare(inkwell::IntPredicate::ULT, a, b, "")
                    .map_err(|e| e.to_string())?;
                let zero = a.get_type().const_zero();
                return builder
                    .build_select(underflow, zero, diff, "")
                    .map_err(|e| e.to_string());
            }
        }
        .map_err(|e| e.to_string())?;
        Ok(r.into())
    }
}

fn emit_cmp<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    op: &CmpOp,
    ty: &ScalarType,
) -> Result<inkwell::values::IntValue<'ctx>, String> {
    if is_float_type(ty) {
        let a = lhs.into_float_value();
        let b = rhs.into_float_value();
        let pred = match op {
            CmpOp::Eq => FloatPredicate::OEQ,
            CmpOp::Ne => FloatPredicate::ONE,
            CmpOp::Lt => FloatPredicate::OLT,
            CmpOp::Le => FloatPredicate::OLE,
            CmpOp::Gt => FloatPredicate::OGT,
            CmpOp::Ge => FloatPredicate::OGE,
        };
        builder
            .build_float_compare(pred, a, b, "cmp")
            .map_err(|e| e.to_string())
    } else {
        let a = lhs.into_int_value();
        let b = rhs.into_int_value();
        let pred = match op {
            CmpOp::Eq => IntPredicate::EQ,
            CmpOp::Ne => IntPredicate::NE,
            CmpOp::Lt => IntPredicate::ULT,
            CmpOp::Le => IntPredicate::ULE,
            CmpOp::Gt => IntPredicate::UGT,
            CmpOp::Ge => IntPredicate::UGE,
        };
        builder
            .build_int_compare(pred, a, b, "cmp")
            .map_err(|e| e.to_string())
    }
}

fn emit_unary<'ctx>(
    builder: &Builder<'ctx>,
    val: BasicValueEnum<'ctx>,
    op: &UnaryOp,
    ty: &ScalarType,
) -> Result<BasicValueEnum<'ctx>, String> {
    match op {
        UnaryOp::Neg => {
            if is_float_type(ty) {
                Ok(builder
                    .build_float_neg(val.into_float_value(), "neg")
                    .map_err(|e| e.to_string())?
                    .into())
            } else {
                Ok(builder
                    .build_int_neg(val.into_int_value(), "neg")
                    .map_err(|e| e.to_string())?
                    .into())
            }
        }
        UnaryOp::BitNot => Ok(builder
            .build_not(val.into_int_value(), "not")
            .map_err(|e| e.to_string())?
            .into()),
        UnaryOp::LogicalNot => Ok(builder
            .build_not(val.into_int_value(), "lnot")
            .map_err(|e| e.to_string())?
            .into()),
    }
}

fn emit_cast<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    val: BasicValueEnum<'ctx>,
    from: &ScalarType,
    to: &ScalarType,
) -> Result<BasicValueEnum<'ctx>, String> {
    let target_ty = scalar_to_llvm_type(context, to);

    match (is_float_type(from), is_float_type(to)) {
        (true, true) => {
            // float -> float (extend or truncate)
            Ok(builder
                .build_float_cast(val.into_float_value(), target_ty.into_float_type(), "fcast")
                .map_err(|e| e.to_string())?
                .into())
        }
        (true, false) => {
            // float -> int
            Ok(builder
                .build_float_to_unsigned_int(
                    val.into_float_value(),
                    target_ty.into_int_type(),
                    "f2i",
                )
                .map_err(|e| e.to_string())?
                .into())
        }
        (false, true) => {
            // int -> float
            Ok(builder
                .build_unsigned_int_to_float(
                    val.into_int_value(),
                    target_ty.into_float_type(),
                    "i2f",
                )
                .map_err(|e| e.to_string())?
                .into())
        }
        (false, false) => {
            // int -> int (extend or truncate)
            Ok(builder
                .build_int_cast(val.into_int_value(), target_ty.into_int_type(), "icast")
                .map_err(|e| e.to_string())?
                .into())
        }
    }
}

pub(crate) fn emit_math_direct<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    arg_vals: &[BasicValueEnum<'ctx>],
    func: &MathFn,
    ty: &ScalarType,
) -> Result<BasicValueEnum<'ctx>, String> {
    let llvm_ty = scalar_to_llvm_type(context, ty);
    let type_suffix = match ty {
        ScalarType::F32 => ".f32",
        ScalarType::F64 => ".f64",
        ScalarType::F16 => ".f16",
        _ => return Err("math functions require float type".into()),
    };

    let intrinsic_name = match func {
        MathFn::Sin => format!("llvm.sin{}", type_suffix),
        MathFn::Cos => format!("llvm.cos{}", type_suffix),
        MathFn::Sqrt => format!("llvm.sqrt{}", type_suffix),
        MathFn::Exp => format!("llvm.exp{}", type_suffix),
        MathFn::Exp2 => format!("llvm.exp2{}", type_suffix),
        MathFn::Log => format!("llvm.log{}", type_suffix),
        MathFn::Log2 => format!("llvm.log2{}", type_suffix),
        MathFn::Pow => format!("llvm.pow{}", type_suffix),
        MathFn::Abs => format!("llvm.fabs{}", type_suffix),
        MathFn::Floor => format!("llvm.floor{}", type_suffix),
        MathFn::Ceil => format!("llvm.ceil{}", type_suffix),
        MathFn::Round => format!("llvm.round{}", type_suffix),
        MathFn::Fma => format!("llvm.fma{}", type_suffix),
        MathFn::Min => format!("llvm.minnum{}", type_suffix),
        MathFn::Max => format!("llvm.maxnum{}", type_suffix),
        // Functions without LLVM intrinsics -- use libdevice or expand
        MathFn::Tan
        | MathFn::Asin
        | MathFn::Acos
        | MathFn::Atan
        | MathFn::Atan2
        | MathFn::Rsqrt
        | MathFn::Clamp => {
            // Fallback: emit as a regular function call (target libdevice provides these)
            format!(
                "__nv_{}{}",
                format!("{:?}", func).to_lowercase(),
                type_suffix
            )
        }
    };

    let fn_type = match arg_vals.len() {
        1 => llvm_ty.fn_type(&[llvm_ty.into()], false),
        2 => llvm_ty.fn_type(&[llvm_ty.into(), llvm_ty.into()], false),
        3 => llvm_ty.fn_type(&[llvm_ty.into(), llvm_ty.into(), llvm_ty.into()], false),
        _ => return Err("math function with unsupported arity".into()),
    };

    let func_val = module
        .get_function(&intrinsic_name)
        .unwrap_or_else(|| module.add_function(&intrinsic_name, fn_type, None));

    let call_args: Vec<inkwell::values::BasicMetadataValueEnum<'ctx>> =
        arg_vals.iter().map(|v| (*v).into()).collect();

    let result = builder
        .build_call(func_val, &call_args, "math")
        .map_err(|e| e.to_string())?
        .try_as_basic_value()
        .basic()
        .ok_or("math function returned void")?;

    Ok(result)
}

/// Create a fixed-width LLVM vector type from a scalar BasicTypeEnum.
fn make_vec_type<'ctx>(scalar: BasicTypeEnum<'ctx>, size: u32) -> VectorType<'ctx> {
    match scalar {
        BasicTypeEnum::FloatType(t) => t.vec_type(size),
        BasicTypeEnum::IntType(t) => t.vec_type(size),
        BasicTypeEnum::PointerType(t) => t.vec_type(size),
        // Structs/arrays/vectors cannot form vector elements in LLVM --
        // this arm should never be reached for valid GPU IR.
        _ => panic!("unsupported scalar type for vector construction"),
    }
}
