//! # Quanta kernel language
//!
//! GPU kernels are written as annotated Rust functions. The `#[quanta::kernel]`
//! proc macro compiles them to GPU code at build time.
//!
//! See `quanta-ir` crate for the full KernelIR specification.
//! See module-level docs for the kernel language reference.

// Re-export IR types from the shared crate
pub use quanta_ir::{
    AtomicOp, BinOp, CmpOp, CompilerOutput, ConstValue, KernelDef, KernelOp, KernelParam, MathFn,
    MatrixFrag, MemoryOrder, Reg, ScalarType, UnaryOp,
};

/// Marker trait for types that can be used in GPU kernels.
pub trait GpuType: Copy + 'static {
    fn gpu_size() -> usize;
    fn scalar_type() -> ScalarType;
}

impl GpuType for f32 {
    fn gpu_size() -> usize {
        4
    }
    fn scalar_type() -> ScalarType {
        ScalarType::F32
    }
}
impl GpuType for f64 {
    fn gpu_size() -> usize {
        8
    }
    fn scalar_type() -> ScalarType {
        ScalarType::F64
    }
}
impl GpuType for u32 {
    fn gpu_size() -> usize {
        4
    }
    fn scalar_type() -> ScalarType {
        ScalarType::U32
    }
}
impl GpuType for i32 {
    fn gpu_size() -> usize {
        4
    }
    fn scalar_type() -> ScalarType {
        ScalarType::I32
    }
}
impl GpuType for u64 {
    fn gpu_size() -> usize {
        8
    }
    fn scalar_type() -> ScalarType {
        ScalarType::U64
    }
}
impl GpuType for i64 {
    fn gpu_size() -> usize {
        8
    }
    fn scalar_type() -> ScalarType {
        ScalarType::I64
    }
}
impl GpuType for u16 {
    fn gpu_size() -> usize {
        2
    }
    fn scalar_type() -> ScalarType {
        ScalarType::U16
    }
}
impl GpuType for i16 {
    fn gpu_size() -> usize {
        2
    }
    fn scalar_type() -> ScalarType {
        ScalarType::I16
    }
}
impl GpuType for u8 {
    fn gpu_size() -> usize {
        1
    }
    fn scalar_type() -> ScalarType {
        ScalarType::U8
    }
}
impl GpuType for i8 {
    fn gpu_size() -> usize {
        1
    }
    fn scalar_type() -> ScalarType {
        ScalarType::I8
    }
}

/// A compiled kernel binary — output of `#[quanta::kernel]` proc macro.
///
/// Contains pre-compiled binaries for each supported GPU vendor.
/// The driver selects the appropriate binary at runtime.
pub struct KernelBinary {
    pub amd: Option<&'static [u8]>,
    pub nvidia: Option<&'static [u8]>,
    pub spirv: Option<&'static [u8]>,
    pub metallib: Option<&'static [u8]>,
    pub wgsl: Option<&'static str>,
}

impl KernelBinary {
    /// Select the best binary for the given vendor.
    /// Apple: metallib binary. Vulkan: SPIR-V. NVIDIA: PTX. AMD: GCN ELF.
    /// Software (CPU): always None — the CPU device only executes the
    /// JIT path (`wave_jit`) on the embedded `KernelDef` IR.
    pub fn for_vendor(&self, vendor: crate::Vendor) -> Option<&[u8]> {
        match vendor {
            crate::Vendor::Amd => self.amd.or(self.spirv),
            crate::Vendor::Nvidia => self.nvidia.or(self.spirv),
            crate::Vendor::Apple => self.metallib,
            crate::Vendor::Intel => self.spirv.or(self.amd),
            crate::Vendor::Software => None,
            _ => self.spirv,
        }
    }
}
