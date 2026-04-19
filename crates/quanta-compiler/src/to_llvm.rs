//! KernelDef → LLVM IR lowering.
//!
//! One emitter for all GPU targets. Target-specific intrinsics
//! (thread IDs, barriers) are provided by the GpuIntrinsics trait.

use std::collections::HashMap;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{CodeModel, FileType, RelocMode, Target, TargetTriple};
use inkwell::types::{BasicType, BasicTypeEnum};
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};

use crate::targets::amdgpu::AmdgpuTarget;
use crate::targets::nvptx::NvptxTarget;
use crate::targets::{GpuIntrinsics, GpuTarget};
use quanta_ir::*;

/// Compile a KernelDef to LLVM IR text for a given GPU target.
pub fn compile_to_llvm_ir(kernel: &KernelDef, target: GpuTarget) -> Result<String, String> {
    let context = Context::create();
    let module = build_module(&context, kernel, target)?;
    Ok(module.print_to_string().to_string())
}

/// Compile a KernelDef to GPU binary for a given target.
/// - NVPTX: returns PTX assembly text (as bytes)
/// - AMDGPU: returns ELF object (as bytes)
pub fn compile_to_binary(kernel: &KernelDef, target: GpuTarget) -> Result<Vec<u8>, String> {
    let context = Context::create();
    let module = build_module(&context, kernel, target)?;

    // Initialize the target
    target.initialize();

    // Create target machine
    let triple = TargetTriple::create(target.triple());
    let llvm_target = Target::from_triple(&triple)
        .map_err(|e| format!("target from triple: {}", e.to_string()))?;

    let opt = match kernel.opt_level {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        _ => OptimizationLevel::Aggressive,
    };

    let target_machine = llvm_target
        .create_target_machine(
            &triple,
            target.cpu(),
            target.features(),
            opt,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or("failed to create target machine")?;

    module.set_data_layout(&target_machine.get_target_data().get_data_layout());

    let pass_name = match kernel.opt_level {
        0 => "default<O0>",
        1 => "default<O1>",
        2 => "default<O2>",
        _ => "default<O3>",
    };
    module
        .run_passes(
            pass_name,
            &target_machine,
            inkwell::passes::PassBuilderOptions::create(),
        )
        .map_err(|e| format!("optimization passes failed: {}", e.to_string()))?;

    // Emit code
    let file_type = match target {
        GpuTarget::Nvptx => FileType::Assembly, // PTX is text assembly
        GpuTarget::Amdgpu => FileType::Object,  // AMD is ELF object
    };

    let buf = target_machine
        .write_to_memory_buffer(&module, file_type)
        .map_err(|e| format!("code emission failed: {}", e.to_string()))?;

    Ok(buf.as_slice().to_vec())
}

fn build_module<'ctx>(
    context: &'ctx Context,
    kernel: &KernelDef,
    target: GpuTarget,
) -> Result<Module<'ctx>, String> {
    let module = context.create_module(&kernel.name);
    let builder = context.create_builder();

    let triple = TargetTriple::create(target.triple());
    module.set_triple(&triple);

    match target {
        GpuTarget::Nvptx => {
            build_kernel(context, &module, &builder, kernel, &NvptxTarget, target)?;
        }
        GpuTarget::Amdgpu => {
            build_kernel(context, &module, &builder, kernel, &AmdgpuTarget, target)?;
        }
    }

    if let Err(msg) = module.verify() {
        return Err(format!("LLVM verification failed: {}", msg.to_string()));
    }

    Ok(module)
}

fn build_kernel<'ctx>(
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
    let mut param_types: Vec<BasicTypeEnum<'ctx>> = Vec::new();
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
            _ => {} // textures — TODO
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

    // Register file — alloca-based (LLVM mem2reg promotes to SSA with phi nodes)
    // This avoids SSA dominance issues when registers are written inside loops/branches.
    let mut reg_slots: HashMap<u32, (PointerValue<'ctx>, ScalarType)> = HashMap::new();

    // Pre-allocate register slots for all registers used in the kernel
    for reg_id in 0..kernel.next_reg {
        let ty = context.f32_type(); // default — will be overwritten on first store
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
            _ => {
                arg_idx += 1;
            }
        }
    }

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
    };
    emit_ops(&mut ectx, &kernel.body)?;

    // Return void
    builder.build_return(None).map_err(|e| e.to_string())?;

    // Add NVPTX kernel metadata
    if target == GpuTarget::Nvptx {
        add_nvptx_kernel_metadata(context, module, &function);
    }

    Ok(())
}

/// Shared context for LLVM IR emission, avoiding excessive parameter passing.
struct EmitCtx<'a, 'ctx> {
    context: &'ctx Context,
    module: &'a Module<'ctx>,
    builder: &'a Builder<'ctx>,
    function: &'a FunctionValue<'ctx>,
    reg_slots: &'a mut HashMap<u32, (PointerValue<'ctx>, ScalarType)>,
    slot_to_arg: &'a HashMap<u32, (PointerValue<'ctx>, ScalarType)>,
    slot_to_const: &'a HashMap<u32, (BasicValueEnum<'ctx>, ScalarType)>,
    intrinsics: &'a dyn GpuIntrinsics<'ctx>,
    _target: GpuTarget,
}

fn emit_ops<'a, 'ctx>(ectx: &mut EmitCtx<'a, 'ctx>, ops: &[KernelOp]) -> Result<(), String> {
    for op in ops {
        emit_op(ectx, op)?;
    }
    Ok(())
}

/// Write a value to a register slot (alloca store).
fn reg_store<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    reg_slots: &mut HashMap<u32, (PointerValue<'ctx>, ScalarType)>,
    reg: u32,
    val: BasicValueEnum<'ctx>,
    ty: ScalarType,
) -> Result<(), String> {
    // If the alloca type doesn't match, create a new one with the right type
    let llvm_ty = scalar_to_llvm_type(context, &ty);
    let need_new = if let Some((_, existing_ty)) = reg_slots.get(&reg) {
        *existing_ty != ty
    } else {
        true
    };
    if need_new {
        let alloca = builder
            .build_alloca(llvm_ty, &format!("r{}", reg))
            .map_err(|e| e.to_string())?;
        reg_slots.insert(reg, (alloca, ty));
    }
    let (ptr, _) = reg_slots.get(&reg).unwrap();
    builder.build_store(*ptr, val).map_err(|e| e.to_string())?;
    Ok(())
}

/// Read a value from a register slot (alloca load).
fn reg_load<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    reg_slots: &HashMap<u32, (PointerValue<'ctx>, ScalarType)>,
    reg: u32,
) -> Result<BasicValueEnum<'ctx>, String> {
    let (ptr, ty) = reg_slots
        .get(&reg)
        .ok_or_else(|| format!("register r{} not allocated", reg))?;
    let llvm_ty = scalar_to_llvm_type(context, ty);
    builder
        .build_load(llvm_ty, *ptr, &format!("r{}", reg))
        .map_err(|e| e.to_string())
}

fn reg_load_int<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    reg_slots: &HashMap<u32, (PointerValue<'ctx>, ScalarType)>,
    reg: &Reg,
) -> Result<IntValue<'ctx>, String> {
    reg_load(context, builder, reg_slots, reg.0).map(|v| v.into_int_value())
}

fn _reg_type(reg_slots: &HashMap<u32, (PointerValue<'_>, ScalarType)>, reg: u32) -> ScalarType {
    reg_slots
        .get(&reg)
        .map(|(_, ty)| *ty)
        .unwrap_or(ScalarType::F32)
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

        _ => {
            // TODO: SharedDecl, SharedLoad, SharedStore, AtomicOp, WaveShuffle, VecConstruct, Texture, Dispatch
        }
    }
    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

fn const_scalar_type(value: &ConstValue) -> ScalarType {
    match value {
        ConstValue::F16(_) => ScalarType::F16,
        ConstValue::F32(_) => ScalarType::F32,
        ConstValue::F64(_) => ScalarType::F64,
        ConstValue::U32(_) => ScalarType::U32,
        ConstValue::U64(_) => ScalarType::U64,
        ConstValue::I32(_) => ScalarType::I32,
        ConstValue::I64(_) => ScalarType::I64,
        ConstValue::Bool(_) => ScalarType::Bool,
    }
}

fn scalar_to_llvm_type<'ctx>(context: &'ctx Context, ty: &ScalarType) -> BasicTypeEnum<'ctx> {
    match ty {
        ScalarType::F16 => context.f16_type().into(),
        ScalarType::F32 => context.f32_type().into(),
        ScalarType::F64 => context.f64_type().into(),
        ScalarType::U8 | ScalarType::I8 => context.i8_type().into(),
        ScalarType::U16 | ScalarType::I16 => context.i16_type().into(),
        ScalarType::U32 | ScalarType::I32 => context.i32_type().into(),
        ScalarType::U64 | ScalarType::I64 => context.i64_type().into(),
        ScalarType::Bool => context.bool_type().into(),
    }
}

fn const_to_llvm<'ctx>(context: &'ctx Context, value: &ConstValue) -> BasicValueEnum<'ctx> {
    match value {
        ConstValue::F32(v) => context.f32_type().const_float(*v as f64).into(),
        ConstValue::F64(v) => context.f64_type().const_float(*v).into(),
        ConstValue::U32(v) => context.i32_type().const_int(*v as u64, false).into(),
        ConstValue::U64(v) => context.i64_type().const_int(*v, false).into(),
        ConstValue::I32(v) => context.i32_type().const_int(*v as u64, true).into(),
        ConstValue::I64(v) => context.i64_type().const_int(*v as u64, true).into(),
        ConstValue::Bool(v) => context.bool_type().const_int(*v as u64, false).into(),
        ConstValue::F16(v) => context
            .f16_type()
            .const_float(f32::from_bits((*v as u32) << 16) as f64)
            .into(),
    }
}

fn is_float_type(ty: &ScalarType) -> bool {
    matches!(ty, ScalarType::F16 | ScalarType::F32 | ScalarType::F64)
}

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
) -> Result<IntValue<'ctx>, String> {
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
            // float → float (extend or truncate)
            Ok(builder
                .build_float_cast(val.into_float_value(), target_ty.into_float_type(), "fcast")
                .map_err(|e| e.to_string())?
                .into())
        }
        (true, false) => {
            // float → int
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
            // int → float
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
            // int → int (extend or truncate)
            Ok(builder
                .build_int_cast(val.into_int_value(), target_ty.into_int_type(), "icast")
                .map_err(|e| e.to_string())?
                .into())
        }
    }
}

fn emit_math_direct<'ctx>(
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
        // Functions without LLVM intrinsics — use libdevice or expand
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
        .left()
        .ok_or("math function returned void")?;

    Ok(result)
}

fn add_nvptx_kernel_metadata<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    function: &FunctionValue<'ctx>,
) {
    // !nvvm.annotations = !{!0}
    // !0 = !{ptr @kernel_name, !"kernel", i32 1}
    let md_string = context.metadata_string("kernel");
    let md_i32 = context.i32_type().const_int(1, false);
    let fn_val = function.as_global_value().as_pointer_value();

    let md_node = context.metadata_node(&[fn_val.into(), md_string.into(), md_i32.into()]);

    module
        .add_global_metadata("nvvm.annotations", &md_node)
        .unwrap_or(());
}
