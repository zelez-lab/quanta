//! NVIDIA NVPTX target — thread intrinsics, barriers, calling convention.

use super::GpuIntrinsics;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::IntValue;

pub struct NvptxTarget;

impl NvptxTarget {
    fn get_sreg<'ctx>(
        &self,
        name: &str,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        let intrinsic_name = format!("llvm.nvvm.read.ptx.sreg.{}", name);
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let func = module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| module.add_function(&intrinsic_name, fn_type, None));
        builder
            .build_call(func, &[], name)
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value()
    }
}

impl<'ctx> GpuIntrinsics<'ctx> for NvptxTarget {
    fn thread_id_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_sreg("tid.x", context, module, builder)
    }

    fn thread_id_y(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_sreg("tid.y", context, module, builder)
    }

    fn block_id_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_sreg("ctaid.x", context, module, builder)
    }

    fn block_dim_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_sreg("ntid.x", context, module, builder)
    }

    fn barrier(&self, context: &'ctx Context, module: &Module<'ctx>, builder: &Builder<'ctx>) {
        let void_type = context.void_type();
        let fn_type = void_type.fn_type(&[], false);
        let func = module
            .get_function("llvm.nvvm.barrier0")
            .unwrap_or_else(|| module.add_function("llvm.nvvm.barrier0", fn_type, None));
        builder.build_call(func, &[], "").unwrap();
    }

    fn kernel_calling_convention(&self) -> u32 {
        // NVPTX doesn't use a special calling convention for kernels.
        // Kernel entry is marked via metadata (!nvvm.annotations).
        0
    }
}
