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
            .basic()
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

    fn wave_shuffle(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        src: IntValue<'ctx>,
        lane_delta: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        // llvm.nvvm.shfl.sync.bfly.i32(mask, val, lane_mask, max_lane_width)
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(
            &[
                i32_type.into(),
                i32_type.into(),
                i32_type.into(),
                i32_type.into(),
            ],
            false,
        );
        let func = module
            .get_function("llvm.nvvm.shfl.sync.bfly.i32")
            .unwrap_or_else(|| module.add_function("llvm.nvvm.shfl.sync.bfly.i32", fn_type, None));
        let mask = i32_type.const_int(0xFFFF_FFFF, false);
        let max_lane = i32_type.const_int(31, false);
        builder
            .build_call(
                func,
                &[mask.into(), src.into(), lane_delta.into(), max_lane.into()],
                "shfl",
            )
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value()
    }

    fn wave_ballot(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        predicate: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        // llvm.nvvm.vote.ballot.sync(mask_i32, pred_i1) -> i32
        let i32_type = context.i32_type();
        let i1_type = context.bool_type();
        let fn_type = i32_type.fn_type(&[i32_type.into(), i1_type.into()], false);
        let func = module
            .get_function("llvm.nvvm.vote.ballot.sync")
            .unwrap_or_else(|| module.add_function("llvm.nvvm.vote.ballot.sync", fn_type, None));
        let mask = i32_type.const_int(0xFFFF_FFFF, false);
        // Truncate predicate to i1 (non-zero = true)
        let pred_i1 = builder
            .build_int_truncate(predicate, i1_type, "pred_i1")
            .unwrap();
        builder
            .build_call(func, &[mask.into(), pred_i1.into()], "ballot")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value()
    }

    fn wave_any(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        predicate: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        // llvm.nvvm.vote.any.sync(mask_i32, pred_i1) -> i1
        let i32_type = context.i32_type();
        let i1_type = context.bool_type();
        let fn_type = i1_type.fn_type(&[i32_type.into(), i1_type.into()], false);
        let func = module
            .get_function("llvm.nvvm.vote.any.sync")
            .unwrap_or_else(|| module.add_function("llvm.nvvm.vote.any.sync", fn_type, None));
        let mask = i32_type.const_int(0xFFFF_FFFF, false);
        let pred_i1 = builder
            .build_int_truncate(predicate, i1_type, "pred_i1")
            .unwrap();
        let result_i1 = builder
            .build_call(func, &[mask.into(), pred_i1.into()], "any")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        // Zero-extend i1 result to i32
        builder
            .build_int_z_extend(result_i1, i32_type, "any_i32")
            .unwrap()
    }

    fn wave_all(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        predicate: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        // llvm.nvvm.vote.all.sync(mask_i32, pred_i1) -> i1
        let i32_type = context.i32_type();
        let i1_type = context.bool_type();
        let fn_type = i1_type.fn_type(&[i32_type.into(), i1_type.into()], false);
        let func = module
            .get_function("llvm.nvvm.vote.all.sync")
            .unwrap_or_else(|| module.add_function("llvm.nvvm.vote.all.sync", fn_type, None));
        let mask = i32_type.const_int(0xFFFF_FFFF, false);
        let pred_i1 = builder
            .build_int_truncate(predicate, i1_type, "pred_i1")
            .unwrap();
        let result_i1 = builder
            .build_call(func, &[mask.into(), pred_i1.into()], "all")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        // Zero-extend i1 result to i32
        builder
            .build_int_z_extend(result_i1, i32_type, "all_i32")
            .unwrap()
    }
}
