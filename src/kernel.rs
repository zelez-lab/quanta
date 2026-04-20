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
    Reg, ScalarType, UnaryOp,
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
    pub msl: Option<&'static str>,
    pub wgsl: Option<&'static str>,
    pub llvm_ir: Option<&'static [u8]>,
}

/// Shader stage — which programmable pipeline stage this shader runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    /// Tessellation control (hull) shader.
    TessControl,
    /// Tessellation evaluation (domain) shader.
    TessEval,
    /// Task (amplification) shader — launches mesh shader threadgroups.
    Task,
    /// Mesh shader — generates vertices and primitives.
    Mesh,
    /// Ray generation shader.
    RayGen,
    /// Closest-hit shader.
    ClosestHit,
    /// Miss shader.
    Miss,
}

/// A compiled shader binary — output of `#[quanta::vertex]` or `#[quanta::fragment]`.
///
/// Contains MSL and WGSL source text compiled at build time from annotated
/// Rust functions. The driver selects the appropriate format at pipeline
/// creation time (MSL for Metal, WGSL for WebGPU/Vulkan).
pub struct ShaderBinary {
    /// Metal Shading Language source.
    pub msl: Option<&'static str>,
    /// WebGPU Shading Language source.
    pub wgsl: Option<&'static str>,
    /// Pre-compiled SPIR-V binary (reserved for future use).
    pub spirv: Option<&'static [u8]>,
    /// Shader entry point name.
    pub entry_point: &'static str,
    /// Shader stage (vertex or fragment).
    pub stage: ShaderStage,
}

impl ShaderBinary {
    /// Select the best shader source for the given vendor.
    ///
    /// Apple: MSL text. Vulkan/WebGPU: WGSL text (or SPIR-V if available).
    pub fn for_vendor(&self, vendor: crate::Vendor) -> Option<&[u8]> {
        match vendor {
            crate::Vendor::Apple => self.msl.map(|s| s.as_bytes()),
            _ => self
                .spirv
                .or(self.wgsl.map(|s| s.as_bytes()))
                .or(self.msl.map(|s| s.as_bytes())),
        }
    }
}

impl KernelBinary {
    /// Select the best binary for the given vendor.
    /// Apple: metallib binary (pre-compiled), fallback to MSL text.
    /// Vulkan: SPIR-V binary. NVIDIA: PTX. AMD: GCN ELF.
    pub fn for_vendor(&self, vendor: crate::Vendor) -> Option<&[u8]> {
        match vendor {
            crate::Vendor::Amd => self.amd.or(self.spirv),
            crate::Vendor::Nvidia => self.nvidia.or(self.spirv),
            crate::Vendor::Apple => self.metallib.or(self.msl.map(|s| s.as_bytes())),
            crate::Vendor::Intel => self.spirv.or(self.amd).or(self.llvm_ir),
            _ => self
                .spirv
                .or(self.wgsl.map(|s| s.as_bytes()))
                .or(self.llvm_ir),
        }
    }
}
