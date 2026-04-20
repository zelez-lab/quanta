//! GPU target backends — target-specific intrinsics and code generation.

pub mod amdgpu;
pub mod nvptx;
pub mod spirv;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::IntValue;

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
#[allow(dead_code)]
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
}
