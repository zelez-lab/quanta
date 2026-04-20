//! SPIR-V target — for Vulkan.
//!
//! Uses LLVM's SPIR-V backend (available in LLVM 19+).
//! Thread intrinsics map to SPIR-V built-in variables.

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::IntValue;

use super::GpuIntrinsics;

pub struct SpirvTarget;

impl SpirvTarget {
    fn get_builtin<'ctx>(
        &self,
        name: &str,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        // SPIR-V uses built-in variables for thread/group IDs.
        // In LLVM SPIR-V, these are accessed via intrinsics similar to NVPTX.
        // For now, use placeholder extern functions — the SPIR-V backend
        // maps them to the correct built-in decorations.
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let func = module
            .get_function(name)
            .unwrap_or_else(|| module.add_function(name, fn_type, None));
        builder
            .build_call(func, &[], name)
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value()
    }
}

impl<'ctx> GpuIntrinsics<'ctx> for SpirvTarget {
    fn thread_id_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_builtin("__spirv_GlobalInvocationId_x", context, module, builder)
    }

    fn thread_id_y(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_builtin("__spirv_GlobalInvocationId_y", context, module, builder)
    }

    fn block_id_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_builtin("__spirv_WorkgroupId_x", context, module, builder)
    }

    fn block_dim_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_builtin("__spirv_WorkgroupSize_x", context, module, builder)
    }

    fn barrier(&self, context: &'ctx Context, module: &Module<'ctx>, builder: &Builder<'ctx>) {
        let void_type = context.void_type();
        let fn_type = void_type.fn_type(&[], false);
        let func = module
            .get_function("__spirv_ControlBarrier")
            .unwrap_or_else(|| module.add_function("__spirv_ControlBarrier", fn_type, None));
        builder.build_call(func, &[], "").unwrap();
    }

    fn kernel_calling_convention(&self) -> u32 {
        // SPIR-V kernels use the default calling convention.
        // The kernel entry point is marked via SPIR-V metadata.
        0
    }
}
