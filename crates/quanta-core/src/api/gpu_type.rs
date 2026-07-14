//! Compute-face kernel types: the `GpuType` marker trait and the
//! `KernelBinary` produced by `#[quanta::kernel]`.
//!
//! These live here (behind `compute`) next to the rest of the compute
//! data model (`Wave`, `Batch`) because `#[quanta::kernel]`-generated
//! code names them, and the companion crates that host kernels must
//! reach them without depending on the `quanta` facade. The facade
//! re-exports both from here, so its public surface is unchanged.
//!
//! `ScalarType` is a `quanta-ir` kernel-language type; it is
//! re-exported alongside these so generated code can name
//! `<crate>::ScalarType` through the same path as `GpuType` /
//! `KernelBinary`.

pub use quanta_ir::ScalarType;

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
///
/// `metallib` is the macOS-platform Metal library. `metallib_ios` /
/// `metallib_ios_sim` are the iOS-device / iOS-simulator variants: iOS
/// rejects a macOS-platform metallib, so a build targeting an iOS device
/// or the simulator embeds and selects its own. The proc macro cannot see
/// the consumer's target (that reaches build scripts only), so it embeds
/// every variant the compiler produced and [`KernelBinary::for_vendor`]
/// picks the platform-correct one by `cfg`.
pub struct KernelBinary {
    pub amd: Option<&'static [u8]>,
    pub nvidia: Option<&'static [u8]>,
    pub spirv: Option<&'static [u8]>,
    pub metallib: Option<&'static [u8]>,
    pub metallib_ios: Option<&'static [u8]>,
    pub metallib_ios_sim: Option<&'static [u8]>,
    pub wgsl: Option<&'static str>,
}

impl KernelBinary {
    /// Select the best binary for the given vendor.
    /// Apple: platform-correct metallib (see [`Self::apple_metallib`]).
    /// Vulkan: SPIR-V. NVIDIA: PTX. AMD: GCN ELF.
    /// Software (CPU): always None — the CPU device only executes the
    /// JIT path (`wave_jit`) on the embedded `KernelDef` IR.
    pub fn for_vendor(&self, vendor: crate::Vendor) -> Option<&[u8]> {
        match vendor {
            crate::Vendor::Amd => self.amd.or(self.spirv),
            crate::Vendor::Nvidia => self.nvidia.or(self.spirv),
            crate::Vendor::Apple => self.apple_metallib(),
            crate::Vendor::Intel => self.spirv.or(self.amd),
            crate::Vendor::Software => None,
            _ => self.spirv,
        }
    }

    /// Resolve the metallib for the current *compile target*.
    ///
    /// Proc macros cannot learn the build target, so all variants are
    /// embedded and the choice is made here by `cfg`. Each target compiles
    /// exactly one arm — the fallback chain (iOS-sim → iOS-device → macOS)
    /// only degrades to a less-specific variant when the more-specific one
    /// wasn't produced. macOS/desktop builds see only the macOS field, so
    /// behavior there is byte-identical to before this method existed
    /// (compute kernels never had a SPIR-V fallback on Apple — a missing
    /// metallib means JIT).
    #[cfg(all(target_os = "ios", target_abi = "sim"))]
    fn apple_metallib(&self) -> Option<&[u8]> {
        self.metallib_ios_sim
            .or(self.metallib_ios)
            .or(self.metallib)
    }

    #[cfg(all(target_os = "ios", not(target_abi = "sim")))]
    fn apple_metallib(&self) -> Option<&[u8]> {
        self.metallib_ios.or(self.metallib)
    }

    #[cfg(not(target_os = "ios"))]
    fn apple_metallib(&self) -> Option<&[u8]> {
        self.metallib
    }
}
