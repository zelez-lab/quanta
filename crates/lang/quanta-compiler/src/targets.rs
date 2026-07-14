//! GPU target backends — target-specific intrinsics and code generation.

pub mod amdgpu;
pub mod nvptx;
pub mod spirv;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{BasicValueEnum, IntValue};

/// Which GPU target to compile for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTarget {
    Nvptx,
    Amdgpu,
    Spirv,
}

impl GpuTarget {
    pub fn triple(&self) -> &'static str {
        match self {
            Self::Nvptx => "nvptx64-nvidia-cuda",
            Self::Amdgpu => "amdgcn-amd-amdhsa",
            Self::Spirv => "spirv64-unknown-unknown",
        }
    }

    pub fn cpu(&self) -> &'static str {
        match self {
            Self::Nvptx => "sm_50",
            Self::Amdgpu => "gfx900",
            Self::Spirv => "",
        }
    }

    pub fn features(&self) -> &'static str {
        match self {
            Self::Nvptx => "+ptx60",
            Self::Amdgpu => "",
            Self::Spirv => "",
        }
    }

    pub fn initialize(&self) {
        use inkwell::targets::{InitializationConfig, Target};
        match self {
            Self::Nvptx => Target::initialize_nvptx(&InitializationConfig::default()),
            Self::Amdgpu => Target::initialize_amd_gpu(&InitializationConfig::default()),
            Self::Spirv => {
                // SPIR-V backend — available in LLVM 19+
                // Try to initialize; if not available, will fail at compilation
                Target::initialize_all(&InitializationConfig::default());
            }
        }
    }
}

/// Target-specific intrinsic helpers.
#[allow(dead_code, clippy::too_many_arguments)]
pub trait GpuIntrinsics<'ctx> {
    fn thread_id_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx>;
    fn thread_id_y(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx>;
    fn block_id_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx>;
    fn block_dim_x(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> IntValue<'ctx>;
    fn barrier(&self, context: &'ctx Context, module: &Module<'ctx>, builder: &Builder<'ctx>);
    fn kernel_calling_convention(&self) -> u32;

    // Wave/subgroup intrinsics
    fn wave_shuffle(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        src: IntValue<'ctx>,
        lane_delta: IntValue<'ctx>,
    ) -> IntValue<'ctx>;

    fn wave_ballot(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        predicate: IntValue<'ctx>,
    ) -> IntValue<'ctx>;

    fn wave_any(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        predicate: IntValue<'ctx>,
    ) -> IntValue<'ctx>;

    fn wave_all(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        predicate: IntValue<'ctx>,
    ) -> IntValue<'ctx>;

    // Texture intrinsics — resolved by the target-specific backend (PTX/AMD/SPIR-V).
    // All targets use extern function stubs; the driver/linker binds them to hardware ops.

    /// Sample a 2D texture at floating-point coordinates. Returns vec4 (<4 x float>).
    fn texture_sample_2d(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        texture_handle: IntValue<'ctx>,
        x: BasicValueEnum<'ctx>,
        y: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx>;

    /// Sample a 3D texture at floating-point coordinates. Returns vec4 (<4 x float>).
    fn texture_sample_3d(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        texture_handle: IntValue<'ctx>,
        x: BasicValueEnum<'ctx>,
        y: BasicValueEnum<'ctx>,
        z: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx>;

    /// Write a vec4 value to a 2D texture at integer coordinates.
    fn texture_write_2d(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        texture_handle: IntValue<'ctx>,
        x: IntValue<'ctx>,
        y: IntValue<'ctx>,
        value: BasicValueEnum<'ctx>,
    );

    /// Query the dimensions of a 2D texture. Returns (width, height) as i32 pair.
    fn texture_size_2d(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        texture_handle: IntValue<'ctx>,
    ) -> (IntValue<'ctx>, IntValue<'ctx>);
}
