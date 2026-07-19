//! Kernel building: parameter mapping, register allocation, entry-point setup.

use std::collections::HashMap;

use inkwell::AddressSpace;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::PointerValue;

use crate::targets::{GpuIntrinsics, GpuTarget};
use quanta_ir::*;

use super::super::metadata::{add_nvptx_kernel_metadata, add_spirv_compute_metadata};
use super::super::{EmitCtx, scalar_to_llvm_type};
use super::ops::emit_ops;

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
            KernelParam::Sampled2D { scalar_type: _, .. }
            | KernelParam::Texture2DRead { scalar_type: _, .. }
            | KernelParam::Texture2DReadWrite { scalar_type: _, .. }
            | KernelParam::Sampled3D { scalar_type: _, .. } => {
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
    let mut slot_to_const: HashMap<u32, (inkwell::values::BasicValueEnum<'ctx>, ScalarType)> =
        HashMap::new();
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
            KernelParam::Sampled2D {
                slot, scalar_type, ..
            }
            | KernelParam::Texture2DRead {
                slot, scalar_type, ..
            }
            | KernelParam::Texture2DReadWrite {
                slot, scalar_type, ..
            }
            | KernelParam::Sampled3D {
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
