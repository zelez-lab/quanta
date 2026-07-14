//! AMD AMDGPU target — thread intrinsics, barriers, calling convention.

use super::GpuIntrinsics;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{BasicValueEnum, IntValue};

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

    fn wave_shuffle(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        src: IntValue<'ctx>,
        lane_delta: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        // llvm.amdgcn.ds.bpermute(byte_offset_i32, src_i32) -> i32
        // byte_offset = lane_id * 4 (ds_bpermute is byte-addressed)
        let i32_type = context.i32_type();
        let fn_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
        let func = module
            .get_function("llvm.amdgcn.ds.bpermute")
            .unwrap_or_else(|| module.add_function("llvm.amdgcn.ds.bpermute", fn_type, None));
        let four = i32_type.const_int(4, false);
        let offset = builder
            .build_int_mul(lane_delta, four, "lane_bytes")
            .unwrap();
        builder
            .build_call(func, &[offset.into(), src.into()], "bpermute")
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
        // llvm.amdgcn.ballot.i64(pred_i1) -> i64, truncate to i32
        let i32_type = context.i32_type();
        let i64_type = context.i64_type();
        let i1_type = context.bool_type();
        let fn_type = i64_type.fn_type(&[i1_type.into()], false);
        let func = module
            .get_function("llvm.amdgcn.ballot.i64")
            .unwrap_or_else(|| module.add_function("llvm.amdgcn.ballot.i64", fn_type, None));
        let pred_i1 = builder
            .build_int_truncate(predicate, i1_type, "pred_i1")
            .unwrap();
        let result_i64 = builder
            .build_call(func, &[pred_i1.into()], "ballot64")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        // Truncate i64 to i32 (lower 32 lanes)
        builder
            .build_int_truncate(result_i64, i32_type, "ballot32")
            .unwrap()
    }

    fn wave_any(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        predicate: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        // wave_any = ballot(pred) != 0
        let i32_type = context.i32_type();
        let i64_type = context.i64_type();
        let i1_type = context.bool_type();
        let fn_type = i64_type.fn_type(&[i1_type.into()], false);
        let func = module
            .get_function("llvm.amdgcn.ballot.i64")
            .unwrap_or_else(|| module.add_function("llvm.amdgcn.ballot.i64", fn_type, None));
        let pred_i1 = builder
            .build_int_truncate(predicate, i1_type, "pred_i1")
            .unwrap();
        let result_i64 = builder
            .build_call(func, &[pred_i1.into()], "ballot64")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let zero_i64 = i64_type.const_int(0, false);
        let cmp = builder
            .build_int_compare(inkwell::IntPredicate::NE, result_i64, zero_i64, "any_cmp")
            .unwrap();
        builder
            .build_int_z_extend(cmp, i32_type, "any_i32")
            .unwrap()
    }

    fn wave_all(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        predicate: IntValue<'ctx>,
    ) -> IntValue<'ctx> {
        // wave_all = ballot(pred) == 0xFFFFFFFFFFFFFFFF (all 64 lanes set)
        let i32_type = context.i32_type();
        let i64_type = context.i64_type();
        let i1_type = context.bool_type();
        let fn_type = i64_type.fn_type(&[i1_type.into()], false);
        let func = module
            .get_function("llvm.amdgcn.ballot.i64")
            .unwrap_or_else(|| module.add_function("llvm.amdgcn.ballot.i64", fn_type, None));
        let pred_i1 = builder
            .build_int_truncate(predicate, i1_type, "pred_i1")
            .unwrap();
        let result_i64 = builder
            .build_call(func, &[pred_i1.into()], "ballot64")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let all_ones = i64_type.const_all_ones();
        let cmp = builder
            .build_int_compare(inkwell::IntPredicate::EQ, result_i64, all_ones, "all_cmp")
            .unwrap();
        builder
            .build_int_z_extend(cmp, i32_type, "all_i32")
            .unwrap()
    }

    fn texture_sample_2d(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        texture_handle: IntValue<'ctx>,
        x: BasicValueEnum<'ctx>,
        y: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        let f32_type = context.f32_type();
        let i32_type = context.i32_type();
        let vec4_type = f32_type.vec_type(4);
        let fn_type =
            vec4_type.fn_type(&[i32_type.into(), f32_type.into(), f32_type.into()], false);
        let func = module
            .get_function("__quanta_tex_sample_2d")
            .unwrap_or_else(|| module.add_function("__quanta_tex_sample_2d", fn_type, None));
        builder
            .build_call(func, &[texture_handle.into(), x.into(), y.into()], "tex2d")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
    }

    fn texture_sample_3d(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        texture_handle: IntValue<'ctx>,
        x: BasicValueEnum<'ctx>,
        y: BasicValueEnum<'ctx>,
        z: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        let f32_type = context.f32_type();
        let i32_type = context.i32_type();
        let vec4_type = f32_type.vec_type(4);
        let fn_type = vec4_type.fn_type(
            &[
                i32_type.into(),
                f32_type.into(),
                f32_type.into(),
                f32_type.into(),
            ],
            false,
        );
        let func = module
            .get_function("__quanta_tex_sample_3d")
            .unwrap_or_else(|| module.add_function("__quanta_tex_sample_3d", fn_type, None));
        builder
            .build_call(
                func,
                &[texture_handle.into(), x.into(), y.into(), z.into()],
                "tex3d",
            )
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
    }

    fn texture_write_2d(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        texture_handle: IntValue<'ctx>,
        x: IntValue<'ctx>,
        y: IntValue<'ctx>,
        value: BasicValueEnum<'ctx>,
    ) {
        let f32_type = context.f32_type();
        let i32_type = context.i32_type();
        let vec4_type = f32_type.vec_type(4);
        let void_type = context.void_type();
        let fn_type = void_type.fn_type(
            &[
                i32_type.into(),
                i32_type.into(),
                i32_type.into(),
                vec4_type.into(),
            ],
            false,
        );
        let func = module
            .get_function("__quanta_tex_write_2d")
            .unwrap_or_else(|| module.add_function("__quanta_tex_write_2d", fn_type, None));
        builder
            .build_call(
                func,
                &[texture_handle.into(), x.into(), y.into(), value.into()],
                "",
            )
            .unwrap();
    }

    fn texture_size_2d(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        texture_handle: IntValue<'ctx>,
    ) -> (IntValue<'ctx>, IntValue<'ctx>) {
        let i32_type = context.i32_type();
        let i64_type = context.i64_type();
        let fn_type = i64_type.fn_type(&[i32_type.into()], false);
        let func = module
            .get_function("__quanta_tex_size_2d")
            .unwrap_or_else(|| module.add_function("__quanta_tex_size_2d", fn_type, None));
        let packed = builder
            .build_call(func, &[texture_handle.into()], "texsize")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        let width = builder
            .build_int_truncate(packed, i32_type, "tex_w")
            .unwrap();
        let shift = i64_type.const_int(32, false);
        let hi = builder
            .build_right_shift(packed, shift, false, "tex_hi")
            .unwrap();
        let height = builder.build_int_truncate(hi, i32_type, "tex_h").unwrap();
        (width, height)
    }
}
