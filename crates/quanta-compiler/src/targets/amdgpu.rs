//! AMD AMDGPU target — thread intrinsics, barriers, calling convention.

use super::GpuIntrinsics;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::IntValue;

pub struct AmdgpuTarget;

impl AmdgpuTarget {
    fn get_intrinsic<'ctx>(
        &self,
        name: &str,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let func = module
            .get_function(name)
            .unwrap_or_else(|| module.add_function(name, fn_type, None));
        builder
            .build_call(func, &[], "")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value()
    }
}

impl<'ctx> GpuIntrinsics<'ctx> for AmdgpuTarget {
    fn thread_id_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_intrinsic("llvm.amdgcn.workitem.id.x", context, module, builder)
    }

    fn thread_id_y(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_intrinsic("llvm.amdgcn.workitem.id.y", context, module, builder)
    }

    fn block_id_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        self.get_intrinsic("llvm.amdgcn.workgroup.id.x", context, module, builder)
    }

    fn block_dim_x(
        &self,
        context: &'ctx Context,
        _module: &Module<'ctx>,
        _builder: &Builder<'ctx>,
    ) -> IntValue<'ctx> {
        // AMDGPU gets block dim from the dispatch packet, not an intrinsic.
        // For now, use a placeholder — real implementation reads from the implicit kernel arg.
        let i32_type = context.i32_type();
        i32_type.const_int(64, false) // default workgroup size
    }

    fn barrier(&self, context: &'ctx Context, module: &Module<'ctx>, builder: &Builder<'ctx>) {
        let void_type = context.void_type();
        let fn_type = void_type.fn_type(&[], false);
        let func = module
            .get_function("llvm.amdgcn.s.barrier")
            .unwrap_or_else(|| module.add_function("llvm.amdgcn.s.barrier", fn_type, None));
        builder.build_call(func, &[], "").unwrap();
    }

    fn kernel_calling_convention(&self) -> u32 {
        91 // AMDGPU_KERNEL calling convention
    }
}
