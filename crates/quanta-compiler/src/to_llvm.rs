//! KernelDef -> LLVM IR lowering.
//!
//! One emitter for all GPU targets. Target-specific intrinsics
//! (thread IDs, barriers) are provided by the GpuIntrinsics trait.

mod emit;
mod metadata;

use std::collections::HashMap;

use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{CodeModel, FileType, RelocMode, Target, TargetTriple};
use inkwell::types::BasicTypeEnum;
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};

use crate::targets::amdgpu::AmdgpuTarget;
use crate::targets::nvptx::NvptxTarget;
use crate::targets::spirv::SpirvTarget;
use crate::targets::{GpuIntrinsics, GpuTarget};
use quanta_ir::*;

use emit::build_kernel;

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
        GpuTarget::Spirv => FileType::Object,   // SPIR-V binary (magic 0x07230203)
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

    // Initialize target and set data layout BEFORE building the kernel.
    // AMDGPU requires addrspace(5) for allocas -- this is encoded in the data layout.
    target.initialize();
    let triple = TargetTriple::create(target.triple());
    module.set_triple(&triple);

    let llvm_target = Target::from_triple(&triple)
        .map_err(|e| format!("target from triple: {}", e.to_string()))?;
    let target_machine = llvm_target
        .create_target_machine(
            &triple,
            target.cpu(),
            target.features(),
            OptimizationLevel::None,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or("failed to create target machine for data layout")?;
    module.set_data_layout(&target_machine.get_target_data().get_data_layout());

    match target {
        GpuTarget::Nvptx => {
            build_kernel(context, &module, &builder, kernel, &NvptxTarget, target)?;
        }
        GpuTarget::Amdgpu => {
            build_kernel(context, &module, &builder, kernel, &AmdgpuTarget, target)?;
        }
        GpuTarget::Spirv => {
            build_kernel(context, &module, &builder, kernel, &SpirvTarget, target)?;
        }
    }

    if let Err(msg) = module.verify() {
        return Err(format!("LLVM verification failed: {}", msg.to_string()));
    }

    Ok(module)
}

// ============================================================================
// Shared types and helpers
// ============================================================================

/// Shared context for LLVM IR emission, avoiding excessive parameter passing.
pub(crate) struct EmitCtx<'a, 'ctx> {
    pub(crate) context: &'ctx Context,
    pub(crate) module: &'a Module<'ctx>,
    pub(crate) builder: &'a Builder<'ctx>,
    pub(crate) function: &'a FunctionValue<'ctx>,
    pub(crate) reg_slots: &'a mut HashMap<u32, (PointerValue<'ctx>, ScalarType)>,
    pub(crate) slot_to_arg: &'a HashMap<u32, (PointerValue<'ctx>, ScalarType)>,
    pub(crate) slot_to_const: &'a HashMap<u32, (BasicValueEnum<'ctx>, ScalarType)>,
    pub(crate) intrinsics: &'a dyn GpuIntrinsics<'ctx>,
    pub(crate) _target: GpuTarget,
}

/// Write a value to a register slot (alloca store).
pub(crate) fn reg_store<'ctx>(
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
pub(crate) fn reg_load<'ctx>(
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

pub(crate) fn reg_load_int<'ctx>(
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

pub(crate) fn scalar_to_llvm_type<'ctx>(
    context: &'ctx Context,
    ty: &ScalarType,
) -> BasicTypeEnum<'ctx> {
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

pub(crate) fn const_to_llvm<'ctx>(
    context: &'ctx Context,
    value: &ConstValue,
) -> BasicValueEnum<'ctx> {
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

pub(crate) fn const_scalar_type(value: &ConstValue) -> ScalarType {
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

pub(crate) fn is_float_type(ty: &ScalarType) -> bool {
    matches!(ty, ScalarType::F16 | ScalarType::F32 | ScalarType::F64)
}
